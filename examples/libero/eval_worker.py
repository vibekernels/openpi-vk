"""Lightweight eval worker — no JAX, no torch at module level.
Called by main_batched.py via spawn. Only imports libero + openpi_client."""

import collections
import math
import os
import pathlib
import sys

import numpy as np


def quat2aa(q):
    q = q.copy()
    if q[3] > 1: q[3] = 1.0
    elif q[3] < -1: q[3] = -1.0
    d = np.sqrt(1 - q[3]*q[3])
    return np.zeros(3) if math.isclose(d, 0) else q[:3] * 2 * math.acos(q[3]) / d


def worker_fn(worker_id, work_items, request_queue, response_queue,
              resize_size, replan_steps, num_steps_wait):
    """Worker process: runs episodes, sends obs for inference, receives actions."""
    _lp = str(pathlib.Path(__file__).resolve().parent.parent.parent / "third_party" / "libero")
    if _lp not in sys.path:
        sys.path.insert(0, _lp)
    os.environ["MUJOCO_GL"] = "egl"

    # Patch torch.load for libero's init_states loading
    import torch
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})

    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools

    DUMMY_ACTION = [0.0] * 6 + [-1.0]
    ENV_RES = 256
    results = []

    for task_id, trial_idx, bddl_path, prompt, init_state, max_steps, seed in work_items:
        env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=ENV_RES, camera_widths=ENV_RES)
        env.seed(seed)
        env.reset()
        obs = env.set_init_state(init_state)
        action_plan = collections.deque()
        t = 0
        success = False

        while t < max_steps + num_steps_wait:
            if t < num_steps_wait:
                obs, _, _, _ = env.step(DUMMY_ACTION)
                t += 1
                continue

            if not action_plan:
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
                wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))
                state = np.concatenate((obs["robot0_eef_pos"], quat2aa(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))

                request_queue.put((worker_id, {
                    "observation/image": img,
                    "observation/wrist_image": wrist,
                    "observation/state": state,
                    "prompt": prompt,
                }))
                actions = response_queue.get()
                action_plan.extend(actions[:replan_steps])

            obs, _, done, _ = env.step(action_plan.popleft().tolist())
            if done:
                success = True
                break
            t += 1

        env.close()
        results.append((task_id, success))

    request_queue.put((worker_id, "DONE", results))
