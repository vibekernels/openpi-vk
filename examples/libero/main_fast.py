"""Fast LIBERO evaluation — single process, one or all tasks.

Parallelism is handled by the shell script (run_fast_eval.sh) which launches
multiple instances of this script, one per task.

No video recording. No tyro (uses argparse).
"""

import argparse
import collections
import dataclasses
import logging
import math
import pathlib
import sys
import time

# Fix libero import
_LIBERO_PATH = str(pathlib.Path(__file__).resolve().parent.parent.parent / "third_party" / "libero")
if _LIBERO_PATH not in sys.path:
    sys.path.insert(0, _LIBERO_PATH)

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _ws

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def run_task(task_id, suite, args):
    task = suite.get_task(task_id)
    init_states = suite.get_task_init_states(task_id)
    task_bddl = str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    max_steps = MAX_STEPS[args.task_suite_name]

    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(args.seed)
    client = _ws.WebsocketClientPolicy(args.host, args.port)

    successes = 0
    for trial in range(args.num_trials_per_task):
        env.reset()
        obs = env.set_init_state(init_states[trial])
        action_plan = collections.deque()
        t = 0
        done = False

        while t < max_steps + args.num_steps_wait:
            try:
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, args.resize_size, args.resize_size))

                if not action_plan:
                    result = client.infer({
                        "observation/image": img,
                        "observation/wrist_image": wrist,
                        "observation/state": np.concatenate((
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )),
                        "prompt": task.language,
                    })
                    action_plan.extend(result["actions"][:args.replan_steps])

                obs, _, done, _ = env.step(action_plan.popleft().tolist())
                if done:
                    successes += 1
                    break
                t += 1
            except Exception as e:
                logging.error(f"Error: {e}")
                break

    env.close()
    return successes, args.num_trials_per_task, task.language


def main():
    logging.basicConfig(level=logging.INFO)

    p = argparse.ArgumentParser()
    p.add_argument("--task-suite-name", default="libero_spatial")
    p.add_argument("--num-trials-per-task", type=int, default=50)
    p.add_argument("--task-id", type=int, default=-1, help="-1 = all tasks")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--resize-size", type=int, default=224)
    p.add_argument("--num-steps-wait", type=int, default=10)
    args = p.parse_args()

    np.random.seed(args.seed)
    start = time.time()

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()

    if args.task_id >= 0:
        task_ids = [args.task_id]
    else:
        task_ids = list(range(suite.n_tasks))

    total_ok = total_n = 0
    for tid in task_ids:
        ok, n, desc = run_task(tid, suite, args)
        total_ok += ok
        total_n += n
        logging.info(f"Task {tid} [{desc}]: {ok}/{n} ({ok/n*100:.0f}%)")

    elapsed = time.time() - start
    # Machine-readable output
    print(f"RESULT|{args.task_suite_name}|{total_ok}|{total_n}|{elapsed:.1f}")


if __name__ == "__main__":
    main()
