"""Collect the payload-estimator calibration dataset.

THE ROUTINE. A fixed, known torque profile -- identical in every episode -- is applied for
a short window while the arm's response is recorded. Because the excitation never varies,
any difference in the response is caused by the arm's state and what it is carrying. A
supervised model then maps (response, state) -> payload mass.

WHY THE UNATTACHED CASE IS IN HERE. The estimator must return a number at ALL times, not
only once something is grasped: ~0 in phase 1, the payload in phase 2, and back to ~0 after
the phase-3 release. So "no payload" is just another label (0.0) rather than a separate
detector. It is collected with the joint OFF and the payload anchored, which is the state
phase 1 is genuinely in -- the arm provably cannot feel the payload there.

WHAT IS RECORDED, and why each is needed:
    tip position / velocity      the raw response to the routine
    payload acceleration         the 'a' in F = m(a - g)
    tip internal force           the 'F' in F = m(a - g); recoverable from strain, which
                                 is what a soft arm's embedded sensors measure
    curvature profile            distributed bending -- the deformation signal
    shape at probe start         reach, mean/peak bend, tip height. WITHOUT THIS the same
                                 routine gives very different responses depending on how
                                 the arm happens to be bent, and the model cannot separate
                                 payload from pose. Measured: adding it halved the error.
Everything is stored RAW per step, so features can be re-derived later without re-running
the expensive part.

MASS FLOOR. Payloads below ~2% of rod mass violate the joint-damping stability limit
(NU_ATTACH * sim_dt), so "no payload" must be genuinely UNATTACHED rather than a tiny one.

Usage:
    python collect_calibration_data.py OUT.npz [n_configs] [seed_offset] [masses_pct]
        masses_pct: comma list, 0 = unattached.  Default 0,5,10,15,20,25,30,35,40,45
Shards can be collected in parallel with different seed_offset and merged later.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import contextlib
import io
import sys
import time

import numpy as np
from set_environment import Environment

OUT = sys.argv[1] if len(sys.argv) > 1 else "data/calib.npz"
N_CFG = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SEED0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
# -1 is a PSEUDO-PAYLOAD meaning "unloaded but BENT": a phase-2 pose with the joint
# switched off. It carries label 0 like the straight unloaded case, and exists because the
# arm after a phase-3 RELEASE is empty AND bent -- a state the straight-rod reset never
# produces. Without it the estimator would extrapolate into the whole post-release regime.
PCTS = ([float(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4
        else [0., -1., 5., 10., 15., 20., 25., 30., 35., 40., 45.])
# RANDOMISED to match how the policies are actually trained: mode 2 redraws the pick
# target every episode and randomize_place redraws the drop-off, which changes the poses
# the phase-2 warm-up produces. Calibrating on fixed ones and deploying on randomised ones
# would feed the estimator configurations it never saw.
MODE = int(os.environ.get("CALIB_MODE", "2"))
RANDOM_PLACE = os.environ.get("CALIB_RANDOM_PLACE", "1") == "1"

PROBE_STEPS = 107                      # ~0.15 s: fits inside the 0.15-0.5 s settle window
N_ACT = 18
AMP = 0.10                             # 10% of actuation range
ROD_MASS = 7.853982
G = np.array([0.0, -9.80665, 0.0])


ROUTINE_HZ = 20.0                      # 3.0 cycles inside the 0.15 s window
PROBE_SECONDS = 0.15


def routine(i, n=PROBE_STEPS, amp=AMP):
    """THE CALIBRATION ROUTINE: a single-frequency sine, identical in every episode.

    FIXED FOREVER. The whole method rests on the excitation never varying, so that any
    difference in the recorded response is caused by the arm and its payload rather than
    by what it was asked to do.

    CHOICE OF SINE. Measured against a step and a 3-40 Hz chirp over the same 60
    configurations with the full feature set: step 5.47, sine 4.58, chirp 4.50 percentage
    points of cross-validated error. Sine and chirp are within noise of one another, so
    the excitation is a SECOND-ORDER choice -- well behind the feature set, where adding
    arm shape and the force/acceleration terms moved the error from ~15 pp to ~4.5 pp.
    A single tone is also the easiest to reproduce on hardware and to interpret: amplitude
    and phase at one frequency are a single point of the arm's transfer function.

    CHOICE OF 20 Hz. The window is 0.15 s, set by settle_min_time -- the guaranteed floor
    of the settle window that follows a grasp, so a probe this long always fits. At the
    12 Hz first considered that window holds only 1.8 cycles, too few to measure amplitude
    and phase reliably; 20 Hz gives 3.0. It still sits well below the loaded joint modes
    (25-57 Hz) and far above the carry dynamics (~0.4 Hz).

    KNOWN RISK of a single tone, recorded because it is why a chirp was considered: one
    frequency can land where two different payloads happen to respond alike, and a sweep
    cannot. If the trained estimator's errors turn out to cluster in specific payload
    PAIRS, that is the signature, and switching back to a sweep is a one-line change.
    """
    t = i / n
    return np.full(N_ACT, amp * np.sin(2 * np.pi * ROUTINE_HZ * t * PROBE_SECONDS))


def build(pct, seed):
    bent_unloaded = (pct < 0)
    rho = 15000.0 * abs(pct) / 100.0 if not bent_unloaded else 3750.0
    attached = pct > 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        e = Environment(
            final_time=5, num_steps_per_update=7, number_of_control_points=6, alpha=75,
            beta=75, COLLECT_DATA_FOR_POSTPROCESSING=False,
            target_position=[-0.4, 0.6, 0.2], target_v=0.5,
            boundary=[-0.6, 0.6, 0.3, 0.9, -0.6, 0.6], place_position=[0.4, 0.05, -0.4],
            place_radius=0.05, place_orient_tol=0.2,
            place_radius_range=(0.50, 0.60),
            floor_height=0.0, E=1e7, sim_dt=2.0e-4, n_elem=20, NU=30, dim=3.5,
            max_rate_of_change_of_activation=np.inf,
            # unattached runs still need a valid payload object; it stays anchored and
            # the joint stays off, so its value cannot reach the arm.
            SPHERE_DENSITY=(rho if attached else 1500.0),
            observe_mass=True,
            # bent-unloaded uses a PHASE-2 reset to get a bent, warmed-up pose, then the
            # joint is switched off below so the arm is genuinely carrying nothing.
            phase=(2 if (attached or bent_unloaded) else 1),
            mode=MODE, randomize_place=RANDOM_PLACE)
        e.reset(seed=seed)
        if bent_unloaded:
            # release the payload: joint off. The arm keeps the bent pose it warmed up
            # into, which is exactly the post-release phase-3 state.
            e.attach_state["attached"] = False
    return e


def collect_one(pct, seed):
    e = build(pct, seed)
    R, S = e.shearable_rod, e.sphere
    dt = e.time_step * e.num_steps_per_update

    shape0 = np.array([
        float(np.linalg.norm(R.position_collection[..., -1])),      # reach
        float(np.linalg.norm(R.kappa, axis=0).mean()),              # mean bend
        float(np.linalg.norm(R.kappa, axis=0).max()),               # peak bend
        float(R.position_collection[1, -1]),                        # tip height
        float(np.linalg.norm(R.velocity_collection[..., -1])),      # tip speed at start
    ])
    centreline0 = R.position_collection.copy()

    # RECORDED GENEROUSLY ON PURPOSE. Re-running the simulation is the expensive part, so
    # everything that might later be wanted for a feature, a plot or a sanity check is
    # stored now: the WHOLE rod, not just its tip, and the payload's own trajectory beside
    # the derived acceleration. Sizes are modest (~80 kB/sample) next to the cost of
    # discovering a missing signal after the fact.
    prev_v = S.velocity_collection[..., 0].copy()
    rec = {k: [] for k in ("rod_p", "rod_v", "tip_p", "tip_v", "tip_dir", "pay_p",
                           "pay_v", "acc", "force", "torque", "kappa", "sigma", "rod_ke")}
    for i in range(PROBE_STEPS):
        with contextlib.redirect_stdout(io.StringIO()):
            e.step(routine(i))
        if not np.all(np.isfinite(R.position_collection)):
            return None
        v = S.velocity_collection[..., 0].copy()
        rec["acc"].append((v - prev_v) / dt); prev_v = v
        rec["rod_p"].append(R.position_collection.copy())          # full centreline
        rec["rod_v"].append(R.velocity_collection.copy())          # full velocity field
        rec["tip_p"].append(R.position_collection[..., -1].copy())
        rec["tip_v"].append(R.velocity_collection[..., -1].copy())
        rec["tip_dir"].append(R.director_collection[..., -1].copy())   # tip orientation
        rec["pay_p"].append(S.position_collection[..., 0].copy())
        rec["pay_v"].append(v)
        rec["force"].append(R.internal_forces[..., -1].copy())      # F in F = m(a - g)
        rec["torque"].append(R.internal_torques[..., -1].copy())    # rotational analogue
        rec["kappa"].append(np.linalg.norm(R.kappa, axis=0).copy())  # bending strain
        rec["sigma"].append(R.sigma.copy())                          # shear/axial strain
        # rod kinetic energy: a scalar that responds strongly to what the arm carries
        rec["rod_ke"].append(0.5 * float(
            (R.mass * (R.velocity_collection ** 2).sum(axis=0)).sum()))
    out = {k: np.array(v, dtype=np.float32) for k, v in rec.items()}
    out.update(
        # -1 is the "unloaded BENT" sentinel; it is a REGIME, not a mass. The label must
        # be 0 like any other unloaded probe -- regime carries the distinction.
        mass_pct=np.float32(max(pct, 0.0)),
        mass_kg=np.float32(max(pct, 0.0) / 100.0 * 7.853982),
        attached=np.int8(1 if pct > 0 else 0),
        # 0 = unloaded straight, 1 = loaded, 2 = unloaded BENT (post-release)
        regime=np.int8(1 if pct > 0 else (2 if pct < 0 else 0)),
        seed=np.int32(seed),
        shape0=shape0.astype(np.float32),
        centreline0=centreline0.astype(np.float32),
        # geometry at probe start, so a plot can show what pose produced each response
        surface_gap0=np.float32(e._surface_gap(float(np.linalg.norm(
            R.position_collection[..., -1] - S.position_collection[..., 0])))),
        body_gap0=np.float32(e._body_gap()),
        tip_cone0=np.float32(e._tip_cone()),
    )
    return out


print(f"routine: {ROUTINE_HZ:.0f} Hz sine, amp {AMP}, {PROBE_STEPS} steps ({PROBE_SECONDS} s, {ROUTINE_HZ*PROBE_SECONDS:.1f} cycles)")
print(f"payloads (% of {ROD_MASS:.3f} kg rod): {PCTS}   (-1 = unloaded BENT)")
print(f"mode={MODE} randomize_place={RANDOM_PLACE}  <- matches how policies are trained")
print(f"{N_CFG} configurations each, seeds {SEED0}..{SEED0 + N_CFG - 1} -> {OUT}\n", flush=True)

samples, t0, diverged = [], time.time(), 0
for pct in PCTS:
    for c in range(N_CFG):
        s = collect_one(pct, SEED0 + c)
        if s is None:
            diverged += 1
            continue
        samples.append(s)
    el = time.time() - t0
    print(f"  {pct:5.1f}%  {len(samples):5d} samples   {el/60:6.1f} min   "
          f"({diverged} diverged)", flush=True)

if not samples:
    print("NO SAMPLES COLLECTED"); sys.exit(1)
out = {k: np.stack([s[k] for s in samples]) for k in samples[0]}
out["routine"] = np.array([routine(i) for i in range(PROBE_STEPS)], dtype=np.float32)
out["probe_steps"] = np.int32(PROBE_STEPS)
out["dt"] = np.float32(2.0e-4 * 7)
out["time"] = (np.arange(PROBE_STEPS, dtype=np.float32) * 2.0e-4 * 7)
# METADATA: everything needed to interpret or reproduce the set without reading this file
_e = build(25.0, SEED0)
out["meta_routine_hz"] = np.float32(ROUTINE_HZ)
out["meta_amp"] = np.float32(AMP)
out["meta_rod_mass"] = np.float32(_e.shearable_rod.mass.sum())
out["meta_rod_radius"] = np.float32(_e.shearable_rod.radius.max())
out["meta_payload_radius"] = np.float32(np.ravel(_e.sphere.radius)[0])
out["meta_n_elem"] = np.int32(_e.shearable_rod.n_elems)
out["meta_sim_dt"] = np.float32(_e.time_step)
out["meta_K_ATTACH"] = np.float32(_e.K_ATTACH)
out["meta_NU_ATTACH"] = np.float32(_e.NU_ATTACH)
out["meta_KT_ATTACH"] = np.float32(_e.KT_ATTACH)
out["meta_NUT_ATTACH"] = np.float32(_e.NUT_ATTACH)
out["meta_gravity"] = np.float32(9.80665)
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
np.savez_compressed(OUT, **out)
print(f"\nSAVED {len(samples)} samples -> {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB, "
      f"{(time.time()-t0)/60:.1f} min, {diverged} diverged)")
print("  keys:", ", ".join(f"{k}{list(v.shape)}" for k, v in out.items()))
