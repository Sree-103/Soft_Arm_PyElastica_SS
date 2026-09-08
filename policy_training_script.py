__doc__ = """This script is to train multiple policies, and or hyper parameter study."""

import os

# Each logging_bio_args.py subprocess below independently lets PyTorch/BLAS spawn a
# thread pool sized to all visible cores. With num_procs processes doing that at once,
# thread counts multiply (5 procs x ~18 threads = ~90 threads on an 8-core machine),
# causing massive oversubscription/context-switch thrashing -- verified: this dropped
# per-process throughput from ~160 fps (solo) to ~4 fps (5-way, unpinned). Capping each
# subprocess to a single thread here (inherited by every child via subprocess.run's
# default env passthrough) lets num_procs processes actually share the cores instead of
# fighting over them.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from multiprocessing import Pool
import subprocess
import sys
from datetime import datetime
import time

# Launch children with THIS interpreter (the .venv_SoftArm python when the script is
# run with it), not whatever bare `python3` resolves to on PATH -- the system python
# has no stable_baselines3 and every seed would die instantly.
PYTHON = sys.executable

# run_onpolicy = True
# run_offpolicy = True
# seed_list = [0, 1, 2, 3, 4]
# batchsize_list_offpolicy = [100000, 200000, 500000, 1000000, 2000000]
# algo_list_offpolicy = ["DDPG", "TD3", "SAC"]
batchsize_list_onpolicy = [1000, 4000, 16000, 64000, 128000]
algo_list_onpolicy = ["PPO", "TRPO"]

run_onpolicy = False                       # off: no PPO/TRPO
run_offpolicy = True
seed_list = [0]                            # seed 0 only for the position-only-release check
batchsize_list_offpolicy = [2000000]       # pick the ONE you want
algo_list_offpolicy = ["SAC"]              # SAC only

tag = "_pickplace_posrelease"              # experiment tag -> model/log filenames

timesteps = 7000000

run_comand_list = []

if run_offpolicy:
    for seed in seed_list:
        for algo in algo_list_offpolicy:
            for batchsize in batchsize_list_offpolicy:
                run_comand = (
                    PYTHON + " logging_bio_args.py"
                    + " --TRAIN"
                    + " --total_timesteps="
                    + str(timesteps)
                    + " --SEED="
                    + str(seed)
                    + " --timesteps_per_batch="
                    + str(batchsize)
                    + " --algo_name="
                    + str(algo)
                    + " --tag="
                    + tag
                )
                run_comand_list.append(run_comand)
                print(run_comand)

if run_onpolicy:
    for seed in seed_list:
        for algo in algo_list_onpolicy:
            for batchsize in batchsize_list_onpolicy:
                run_comand = (
                    PYTHON + " logging_bio_args.py"
                    + " --TRAIN"
                    + " --total_timesteps="
                    + str(timesteps)
                    + " --SEED="
                    + str(seed)
                    + " --timesteps_per_batch="
                    + str(batchsize)
                    + " --algo_name="
                    + str(algo)
                    + " --tag="
                    + tag
                )
                run_comand_list.append(run_comand)
                print(run_comand)


def run_command_fun(command):
    print(command)
    print("command started at:", datetime.now())
    start = time.time()
    subprocess.run(command, shell=True)
    # time.sleep(1)
    done = time.time()
    print(
        "command finished at:",
        datetime.now(),
        "It took:",
        (done - start) / 60,
        "minutes",
    )
    print()


num_procs = (
    5  # make smaller than the number of cores to take advantage of multiple threads
)
pool = Pool(num_procs)
pool.map(run_command_fun, run_comand_list)