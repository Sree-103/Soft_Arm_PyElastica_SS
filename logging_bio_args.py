__doc__ = """This script is to train or run a policy for the arm reaching to a fixed target with specific orientation.
Case 2 in CoRL 2020 paper."""
import os

# Cap every math library to ONE thread, and do it HERE -- before numpy/numba/torch are
# imported, because OpenBLAS/OMP read these at library load. policy_training_script.py
# sets the same caps, but baking them in here protects direct launches too: 5 unpinned
# processes on 8 cores previously collapsed throughput from ~160 fps to ~4 fps.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import sys

import argparse
import matplotlib
import matplotlib.pyplot as plt

# Import stable baselines 3
# (stable_baselines TF1 does not exist for Python 3.12; per-algorithm MlpPolicy
# imports are gone in SB3 -- the policy is the string "MlpPolicy" for all algos)
from stable_baselines3.common.monitor import Monitor, load_results
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import DDPG, PPO, TD3, SAC

# torch honors OMP_NUM_THREADS, but pin it explicitly as well so a preloaded/odd build
# cannot re-expand its intra-op pool under 5-way concurrency.
import torch

torch.set_num_threads(1)

# Import simulation environment
from set_environment import Environment


def get_valid_filename(s):
    import re

    s = str(s).strip().replace(" ", "_")
    return re.sub(r"(?u)[^-\w.]", "", s)


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results
    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.savefig(title + ".png")
    plt.close()


parser = argparse.ArgumentParser()

########### training and data info ###########
parser.add_argument(
    "--total_timesteps", type=float, default=10000000,
)

parser.add_argument(
    "--SEED", type=int, default=0,
)

parser.add_argument(
    "--timesteps_per_batch", type=int, default=2000000,
)

parser.add_argument(
    "--algo_name", type=str, default="SAC",
)
parser.add_argument(
    "--tag", type=str, default="",
)
parser.add_argument(
    "--TRAIN", action="store_true",
    help="If passed, train. Otherwise run the trained policy.",
)

parser.add_argument(
    "--action-rate-penalty", type=float, default=0.0,
    help="Weight w on the PHASE-2 action-rate penalty, w * mean((a_t - a_{t-1})^2), "
         "subtracted from R2 only. 0.0 (default) = off, identical to every run so far. "
         "Calibration: the measured mean square action change is 0.256/step against a "
         "typical step reward of 3.99, so w=1.0 costs ~6.4%% while the inner->outer tier "
         "gap is 2.0 -- tracking therefore stays strictly dominant and the policy only "
         "buys smoothness where it is free. Nothing else in the reward changes.",
)
parser.add_argument(
    "--action-penalty", type=float, default=0.0,
    help="Weight w on the PHASE-2 action-MAGNITUDE penalty, w * mean(a_t^2), subtracted "
         "from R2 only. 0.0 (default) = off. INDEPENDENT of --action-rate-penalty: they "
         "tax different things (effort vs jitter) and may be used together. Calibration: "
         "measured mean ||a||^2/d is 0.563 against mean ||da||^2/d of 0.239, so the same "
         "weight costs 2.4x more here; w=0.3 matches the ~6%% budget the rate penalty was "
         "sized against, while w=1.0 would take ~21%% of the step reward.",
)
parser.add_argument(
    "--sphere-density-set", type=str, default=None,
    help="Comma-separated payload densities to draw from at EVERY reset, e.g. "
         "'100,300,500'. Unset = the single fixed SPHERE_DENSITY (100), exactly as every "
         "run so far. mass = density * 5.236e-4 kg at the fixed 0.05 m radius, so "
         "100/300/500 = 0.052/0.157/0.262 kg. The draw is per EPISODE (constant within "
         "one episode) and uses the env's np_random, so it is reproducible from --SEED "
         "and each seed gets its own order. Radius is deliberately NOT varied: it sets "
         "the drop-off height, the attach gate distance and the on-screen size, so "
         "changing it would move the task rather than just the payload.",
)
parser.add_argument(
    "--mass-estimator", type=str, default=None,
    help="Path to a frozen estimator .npz from train_estimator.py. When given, "
         "observation entry 56 carries the ESTIMATED payload from the calibration routine "
         "instead of the true mass, and the true value never reaches the policy. Mutually "
         "exclusive with --observe-mass, which is the ORACLE baseline.")
parser.add_argument(
    "--train-freq", type=int, default=1,
    help="SAC gradient updates are performed every N environment steps. SB3's default of "
         "1 pays the per-update overhead on EVERY step, and that update costs ~4.0 ms "
         "against ~1.5 ms for the simulation -- 73%% of wall clock. The update is "
         "OVERHEAD-bound, not compute-bound (batch 256 costs only 26%% more than batch 64, "
         "and extra threads do nothing), so raising this while raising --batch-size in "
         "proportion keeps the same number of replay samples processed per environment "
         "step while paying the overhead far less often. Measured: train_freq 4 with batch "
         "256 is ~2x faster in wall clock at identical sample throughput.")
parser.add_argument(
    "--batch-size", type=int, default=64,
    help="SAC minibatch. Raise together with --train-freq to hold samples-per-env-step "
         "constant. NOTE the divergence warning in the code refers to batch 256 WITH "
         "net_arch [256,256]; the network stays [64,64] here.")
parser.add_argument(
    "--observe-mass", action="store_true",
    help="Append the payload mass (as mass/rod_mass, dimensionless) to the observation. "
         "CHANGES THE OBSERVATION DIMENSION 55 -> 56, so a policy trained with this flag "
         "cannot be loaded by a run without it, and every phase must be retrained "
         "together. Off by default so all existing policies stay loadable.",
)
parser.add_argument(
    "--track-tier-inner", type=float, default=None,
    help="PHASE-2 inner tracking tier. Default None = place_radius (0.05), so the tier "
         "EQUALS the arrival gate, mirroring phase 1 where the inner tier equals "
         "attach_radius. Set 0.08 as the documented fallback if 0.05 proves too sparse.",
)
parser.add_argument(
    "--track-tier-outer", type=float, default=None,
    help="PHASE-2 outer tracking tier. Default None = 2*place_radius (0.10).",
)
parser.add_argument(
    "--fixed-place", action="store_true",
    help="Pin the drop-off instead of drawing it from the annulus every episode. For fast "
         "diagnostic runs only -- it makes the task much easier, so a run using it can only "
         "be compared against another run that also uses it.",
)
parser.add_argument(
    "--orient-tier-inner", type=float, default=0.20,
    help="PHASE-3 inner downward-posture tier AND the release tolerance on p_down "
         "(p_down = 0 is straight down; tier t is acos(1-2t) deg off vertical, so 0.20 "
         "= 53 deg, 0.15 = 46, 0.12 = 41, 0.10 = 37). Default 0.20 reproduces every run "
         "so far. TIGHTEN IT TO GIVE PHASE 3 A JOB: measured on the recorded arrivals, "
         "the release gate is ALREADY satisfied on arrival for 74-76%% of episodes at "
         "0.20, 63-64%% at 0.15, 53-55%% at 0.12 and 44-47%% at 0.10 -- the distance half "
         "of the gate is 100%% pre-satisfied by construction, since phase 2 ends at the "
         "arrival latch, so orientation is the only thing phase 3 can be trained on. "
         "Against that, feasibility falls: of 2982 measured loaded floor poses in the "
         "0.50-0.60 m annulus, 78%% can point down at 0.20, 67%% at 0.15 and 61%% at 0.10, "
         "so tightening also lowers the achievable success ceiling.",
)
parser.add_argument(
    "--orient-tier-outer", type=float, default=0.35,
    help="PHASE-3 outer downward-posture tier (bonus only, not the release gate). "
         "Default 0.35 reproduces every run so far.",
)
parser.add_argument(
    "--phase", type=int, default=0, choices=[0, 1, 2, 3],
    help="Three-policy split: 1 = policy 1 (reach+orient+grasp, episode ends at the "
         "grasp), 2 = policy 2 (loaded carry along the dome, episode starts already "
         "grasped and ENDS at the drop-off -- it can no longer release), 3 = policy 3 "
         "(orient the tip down at the drop-off, then release; episode starts there "
         "via its own warm-up), 0 = combined single policy. Each phase builds its own "
         "start state, so any one can be retrained without the others.",
)
parser.add_argument(
    "--record-arrival-states", type=str, default=None,
    help="Path to WRITE arrival states to while phase 2 trains. Passive. Feed the "
         "result to phase 3 via --phase3-arrival-states.",
)
parser.add_argument(
    "--phase3-arrival-states", type=str, default=None,
    help="Path to a .npz of REAL policy-2 arrival states. When given, phase 3 restores "
         "one per episode instead of synthesising a start, and place_position stays the "
         "REAL drop-off instead of being moved under a flailed rod.",
)
parser.add_argument(
    "--record-attach-states", type=str, default=None,
    help="Path to WRITE attach states to while phase 1 trains. Passive -- touches no "
         "reward or gate, so phase-1 training is unchanged. Feed the resulting file to "
         "phase 2 via --phase2-attach-states.",
)
parser.add_argument(
    "--phase2-attach-states", type=str, default=None,
    help="Path to a .npz of REAL policy-1 attach states (harvest_attach_states.py). "
         "When given, every phase-2 episode RESTORES one instead of synthesising a "
         "grasp, so policy 2 trains on the hand-over dynamics it actually inherits.",
)
parser.add_argument(
    "--phase2-policy1", type=str, default=None,
    help="Path to the trained policy 1 (no .zip). When given, every phase-2 episode "
         "starts from a REAL policy-1 grasp instead of the synthetic warm-up, so "
         "policy 2 trains on the true hand-over distribution. Optional.",
)
parser.add_argument(
    "--phase3-policy2", type=str, default=None,
    help="Path to a trained policy 2 (no .zip). OPTIONAL -- by default phase 3 builds "
         "its own start state and needs no policy 2. Give this to seed phase-3 episodes "
         "from REAL policy-2 arrivals instead, trading independence for fidelity.",
)

args = parser.parse_args()

# SB3: the policy is the string "MlpPolicy" for every algorithm.
MLP = "MlpPolicy"

if args.algo_name == "TRPO":
    # TRPO is not in core SB3; it lives in sb3_contrib (pip install sb3-contrib).
    from sb3_contrib import TRPO

    algo = TRPO
    batchsize = "n_steps"  # SB3 name for the on-policy rollout size
    offpolicy = False
elif args.algo_name == "PPO":
    algo = PPO  # PPO1 -> PPO in SB3
    batchsize = "n_steps"  # SB3 name (was timesteps_per_actorbatch)
    offpolicy = False
elif args.algo_name == "DDPG":
    algo = DDPG
    batchsize = "nb_rollout_steps"
    offpolicy = True
elif args.algo_name == "TD3":
    algo = TD3
    batchsize = "train_freq"
    offpolicy = True
elif args.algo_name == "SAC":
    algo = SAC
    batchsize = "train_freq"
    offpolicy = True

# Mode 2 corresponds to fixed rotated fixed target
# MODE=1 pins BOTH the pick target and its orientation (target_position as configured
# below, theta_y = pi/4) instead of redrawing them from the boundary box every episode.
# For deterministic diagnostic runs only -- it removes the reach randomisation the paper's
# Case 2 is defined by, so a run using it cannot be compared with a mode-2 run. Verified
# that mode 1 changes nothing else: every other self.mode branch in set_environment.py is
# for modes 3/4 (moving targets), which this does not touch.
args.mode = int(os.environ.get("MODE", "2"))
if args.mode != 2:
    print(f"MODE={args.mode} -- pick target and its orientation are FIXED (diagnostic run)")

# Set simulation final time. 2 -> 5 for the DOME carry: the phase-2 goal rides the
# dome at carry_speed 0.5 m/s, so a typical ~1.3-1.6 m dome takes ~2.6-3.2 s AFTER
# the attach (itself ~0.5-1 s in); 2 s could never finish a carry, 5 s leaves ~1 s
# of settled-placement time that R2 keeps scoring.
# PHASE 2 also needs 5 s. It starts grasped, but the warm-up that places the grasp in
# the real hand-over range costs up to 0.60 s, and the dome from the far side of that
# range can run ~1.9 m = 3.8 s at carry_speed 0.5. 4 s would truncate the longest
# carries before the goal ever reached the drop-off, so no release could fire on them.
final_time = 5
# Number of control points
number_of_control_points = 6
# target position
target_position = [-0.4, 0.6, 0.2]
# learning step skip
num_steps_per_update = 7

# Pick-and-place: fixed, constant drop-off location (never randomized, independent of
# mode/boundary). y MUST equal floor_height + sphere_radius = 0.05 so the placed object
# RESTS at place_position after release (set_environment.py docstring: a mid-air
# place_position makes the object fall away post-release, R2 goes negative, and the
# agent learns to never release). Below the pick-sampling box (pick y in [0.3, 0.9]) --
# a "carry it down to the ground" placement task, well within the 1.0 m arm reach.
# Dome END point = the SAME drop-off the _pickplace_posrelease run used and placed
# successfully. It was moved to [0.4, 0.05, -0.5] for the first dome run; that put it
# 0.642 m from the base instead of 0.568 m, and measurements on the resulting policies
# showed every carry completing (s reached 1.000) but stopping a median 0.10 m short of
# a 0.05 m gate -- i.e. the extra 7.4 cm at floor level exhausted the loaded arm's reach
# margin. Reverting also makes the dome the ONLY change vs posrelease, so the two runs
# are directly comparable.
place_position = [0.4, 0.05, -0.4]

# alpha and beta spline scaling factors in normal/binormal and tangent directions respectively
args.alpha = 75
args.beta = 75

sim_dt = 2.0e-4

# ACTUATOR SLEW LIMIT, now FINITE. The muscle activation may change by at most this much
# PER SIM STEP (filter_activation runs inside apply_torques, which PyElastica calls every
# step -- NOT once per control step). Activations live in [-1, 1], so 2.0 would permit a
# full-range jump in one 2e-4 s step and be identical to inf; the useful range is well
# below that.
#
# 0.05/sim-step = 0.35 per control step (7 sim steps), against a measured natural slew of
# ~0.49 per component per control step -- so it binds, cutting the peak command rate to
# about 70% of what the unconstrained policy used.
#
# This replaces the reward-side action penalties, which are now commented out in R2:
# smoothing is done in the PLANT rather than paid for in the reward, so the policy is not
# charged twice for the same behaviour. Override with SLEW_LIMIT.
max_rate_of_change_of_activation = float(os.environ.get("SLEW_LIMIT", "0.05"))
print("rate of change", max_rate_of_change_of_activation)

env = Environment(
    final_time=final_time,
    num_steps_per_update=num_steps_per_update,
    number_of_control_points=number_of_control_points,
    alpha=args.alpha,
    beta=args.beta,
    COLLECT_DATA_FOR_POSTPROCESSING=not args.TRAIN,
    mode=args.mode,
    target_position=target_position,
    target_v=0.5,
    boundary=[-0.6, 0.6, 0.3, 0.9, -0.6, 0.6],
    place_position=place_position,   # required: fixed pick-and-place target
    place_radius=0.05,               # matches attach_radius (equals the class default)
    # RANDOMISED DROP-OFF, the phase-2 analogue of phase 1's randomised pick target.
    # Sampled per episode from an ANNULUS on the floor -- not a box, because at floor
    # level the loaded arm is near its envelope. Band from map_dropoff_workspace.py:
    # of 2982 loaded floor-level poses, the fraction that could ALSO point down was
    # 0% (0.3-0.4 m), 34% (0.4-0.5), 67% (0.5-0.6), 28% (0.6-0.7), 4% (0.7-0.8),
    # 0% (0.8-0.9). Points outside 0.5-0.6 m would spawn drop-offs phase 3 cannot
    # release at. Set randomize_place=False to return to the single fixed drop-off.
    randomize_place=not args.fixed_place,
    place_radius_range=(0.50, 0.60),
    # DEAD PARAMETER, kept only so the call signature does not change. It used to
    # gate the release on a quaternion distance; the phase-3 release now tests
    # p_down (a tip DIRECTION error) against orient_tier_inner below, and nothing
    # reads place_orient_tol at all. Tune orient_tier_inner, not this.
    place_orient_tol=0.2,
    # PHASE-3 downward-posture tiers, passed explicitly so the training config is
    # self-documenting rather than relying on the env defaults. p_down = 0 is the
    # tip pointing straight down; 0.15 is 46 deg off vertical, 0.30 is 66 deg.
    # orient_tier_inner is BOTH the inner bonus tier and the release tolerance.
    # Calibrated from the five hand-off episodes, attached frames only: the best
    # p_down reached while inside place_radius was 0.027/0.409/0.020/0.069/0.111,
    # Loosened 0.15 -> 0.20 now that the drop-off is randomised: of the poses inside
    # the 0.50-0.60 m spawn annulus, 67% can point down at 0.15 but 78% at 0.20, and
    # nothing more is gained past 0.20 (the rest cannot be oriented at any tolerance).
    # Now settable from the CLI (--orient-tier-outer / --orient-tier-inner). The
    # defaults below are the values every run so far used, so omitting the flags
    # reproduces those runs exactly.
    orient_tier_outer=args.orient_tier_outer,
    orient_tier_inner=args.orient_tier_inner,
    # Phase-2 action-rate penalty (see --action-rate-penalty). 0.0 = off.
    action_rate_penalty=args.action_rate_penalty,
    # Phase-2 action-magnitude penalty (see --action-penalty). 0.0 = off.
    action_penalty=args.action_penalty,
    # Payload mass in the observation (see --observe-mass). 55 -> 56 dims.
    observe_mass=args.observe_mass,
    mass_estimator=args.mass_estimator,
    # Tracking tiers. None -> the env derives them from place_radius (0.10 / 0.05).
    **({"track_tier_inner": args.track_tier_inner} if args.track_tier_inner else {}),
    **({"track_tier_outer": args.track_tier_outer} if args.track_tier_outer else {}),
    # Payload mass randomisation (see --sphere-density-set). None = fixed density.
    sphere_density_set=(
        [float(x) for x in args.sphere_density_set.split(",")]
        if args.sphere_density_set else None
    ),
    floor_height=0.0,                # ground level for FloorForSphere (class default)
    E=1e7,
    sim_dt=sim_dt,
    n_elem=20,
    NU=30,
    dim=3.5,
    max_rate_of_change_of_activation=max_rate_of_change_of_activation,
    phase=args.phase,                # 1 = reach, 2 = carry, 3 = place, 0 = combined
    record_arrival_states=args.record_arrival_states,
    phase3_arrival_states=args.phase3_arrival_states,
    record_attach_states=args.record_attach_states,
    phase2_attach_states=args.phase2_attach_states,
    phase2_policy1=args.phase2_policy1,
    phase3_policy2=args.phase3_policy2,
)

name = str(args.algo_name) + "_3d-tracking_id"
identifer = name + "-" + str(args.timesteps_per_batch) + "_tt-" + str(int(args.total_timesteps)) + args.tag + "_" + str(args.SEED)

if args.TRAIN:
    log_dir = "./log_" + identifer + "/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)


from stable_baselines3.common.results_plotter import ts2xy, plot_results
from stable_baselines3.common import results_plotter

if args.TRAIN:
    if offpolicy:
        if args.algo_name == "TD3":
            items = {
                "policy": MLP,
                "buffer_size": int(args.timesteps_per_batch),
                "learning_starts": int(50e3),
                # Match the TF1 stable-baselines SAC/TD3 defaults that trained Case 2:
                # network [64,64] and batch 64. SB3 defaults ([256,256], batch 256)
                # give the critic enough capacity to chase the sparse reward spikes
                # into divergence (verified: critic_loss climbs unbounded otherwise).
                "policy_kwargs": dict(net_arch=[64, 64]),
                "batch_size": 64,
            }
        else:
            items = {
                "policy": MLP,
                "buffer_size": int(args.timesteps_per_batch),
                # ~7 episodes (1428 steps each) of warm-up data before the first
                # gradient update, instead of the SB3 default of 100 steps -- early
                # updates otherwise fit noise from a single trajectory.
                "learning_starts": 10000,
                # Match the TF1 stable-baselines SAC defaults that trained Case 2:
                # network [64,64] and batch 64. SB3 defaults ([256,256], batch 256)
                # give the critic enough capacity to chase the sparse reward spikes
                # into divergence (verified: critic_loss climbs unbounded otherwise).
                "policy_kwargs": dict(net_arch=[64, 64]),
                "batch_size": int(args.batch_size),
                "train_freq": int(args.train_freq),
                # one update per train_freq steps; keep the ratio to batch constant
                "gradient_steps": 1,
            }
    else:
        items = {
            "policy": MLP,
            batchsize: args.timesteps_per_batch,
        }

    model = algo(env=env, verbose=1, seed=args.SEED, **items)
    model.set_env(env)

    # Periodic snapshots. The previous run saved ONLY the final weights, and seed 2
    # was still releasing on 32% of its grasps in its last 100 training episodes while
    # its saved snapshot released 0/158 at test -- with no checkpoints there was no way
    # to recover the policy that had been working. Checkpoints are pure I/O: they touch
    # no reward, no environment and no learning dynamics.
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000,                      # ~14 snapshots over 7M, ~6 over 3M
        save_path="./ckpt_" + identifer + "/",
        name_prefix="policy",
    )
    model.learn(total_timesteps=int(args.total_timesteps), callback=checkpoint_callback)

    # SAVE FIRST: a plotting failure must never cost a trained policy.
    model.save("policy-" + identifer)

    try:
        # library helper
        plot_results(
            [log_dir],
            int(args.total_timesteps),
            results_plotter.X_TIMESTEPS,
            str(args.algo_name) + "_" + identifer,
        )
        plt.savefig("convergence_plot" + identifer + ".png")
    except Exception as e:
        print("Plotting failed (model already saved):", e)

else:
    # Use trained policy for the simulation.
    model = SAC.load("policy-" + identifer)

    # SB3 load() re-applies the training seed stored inside the model, which
    # reseeds numpy/torch globally -- every playback would replay the exact
    # same "random" targets and sampled actions. Re-randomize so each test
    # run draws fresh episodes.
    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(int.from_bytes(os.urandom(4), "little"))

    for _ in range(9):
        obs, _ = env.reset()  # gymnasium: reset returns (obs, info)

    done = False
    score = 0
    while not done:
        # deterministic=False: the mean action learned to hover just outside the
        # attach gate (touch-but-don't-trigger pays better than carrying under
        # the v1 reward switch); the stochastic policy crosses the gate at the
        # training-time rate, so playback samples actions to make attach/carry
        # episodes observable.
        action, _states = model.predict(obs, deterministic=False)
        # gymnasium: step returns 5-tuple
        obs, rewards, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += rewards
        if info["ctime"] > final_time:
            break
    print("Final Score:", score)
    env.post_processing(
        filename_video="video-" + identifer + ".mp4", SAVE_DATA=True,
    )