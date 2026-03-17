"""Batched parallel LIBERO evaluation.

Architecture:
- N worker processes (spawned), each with own mujoco env
- Workers send observations via queue to coordinator
- Coordinator batches observations, runs one GPU inference call, sends results back
- Workers receive actions and continue stepping
"""

import argparse
import collections
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

# Only import JAX/model in main process (workers use eval_worker.py)
_LIBERO_PATH = str(pathlib.Path(__file__).resolve().parent.parent.parent / "third_party" / "libero")

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-suite-name", default="libero_spatial")
    p.add_argument("--num-trials-per-task", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--resize-size", type=int, default=224)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--checkpoint-dir", default="/workspace/checkpoints/pi05_libero_lora/optimized_run/4999")
    p.add_argument("--config-name", default="pi05_libero_lora")
    p.add_argument("--batch-timeout", type=float, default=0.005)
    args = p.parse_args()

    np.random.seed(args.seed)

    # ---- Load model (main process only) ----
    import jax
    import jax.numpy as jnp
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    from openpi.models import model as _model

    t0 = time.time()
    config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(config, args.checkpoint_dir)

    sample_actions = policy._sample_actions
    sample_kwargs = dict(policy._sample_kwargs)
    input_transform = policy._input_transform
    output_transform = policy._output_transform

    # JIT warmup
    dummy = {
        "observation/image": np.zeros((args.resize_size, args.resize_size, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((args.resize_size, args.resize_size, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float64),
        "prompt": "test",
    }
    policy.infer(dummy)
    load_time = time.time() - t0
    print(f"Model loaded + JIT warmup: {load_time:.1f}s", flush=True)

    def batched_infer(obs_list):
        nonlocal policy
        if len(obs_list) == 1:
            r = policy.infer(obs_list[0])
            return [r["actions"]]

        transformed = []
        for obs in obs_list:
            inp = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in obs.items()}
            inp = input_transform(inp)
            transformed.append(inp)

        batched = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *transformed)
        policy._rng, sample_rng = jax.random.split(policy._rng)
        observation = _model.Observation.from_dict(batched)
        actions = sample_actions(sample_rng, observation, **sample_kwargs)

        results = []
        for i in range(len(obs_list)):
            out = {"state": np.asarray(batched["state"][i]), "actions": np.asarray(actions[i])}
            out = output_transform(out)
            results.append(out["actions"])
        return results

    # ---- Build work items ----
    if _LIBERO_PATH not in sys.path:
        sys.path.insert(0, _LIBERO_PATH)

    import torch
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})

    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    max_steps = MAX_STEPS[args.task_suite_name]

    all_work = []
    for tid in range(suite.n_tasks):
        task = suite.get_task(tid)
        init_states = suite.get_task_init_states(tid)
        bddl = str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
        for trial in range(args.num_trials_per_task):
            all_work.append((tid, trial, bddl, task.language, init_states[trial], max_steps, args.seed))

    total_ep = len(all_work)
    n_workers = min(args.num_workers, total_ep)

    # Distribute work round-robin
    worker_work = [[] for _ in range(n_workers)]
    for i, item in enumerate(all_work):
        worker_work[i % n_workers].append(item)

    print(f"Suite: {args.task_suite_name} | {total_ep} episodes | {n_workers} workers", flush=True)

    # ---- Launch workers ----
    from examples.libero.eval_worker import worker_fn

    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue()
    response_queues = [ctx.Queue() for _ in range(n_workers)]

    workers = []
    for wid in range(n_workers):
        pw = ctx.Process(
            target=worker_fn,
            args=(wid, worker_work[wid], request_queue, response_queues[wid],
                  args.resize_size, args.replan_steps, args.num_steps_wait),
        )
        pw.start()
        workers.append(pw)

    # ---- Coordinator loop ----
    eval_start = time.time()
    n_infer_calls = 0
    n_batches = 0
    workers_done = 0
    all_results = []

    while workers_done < n_workers:
        batch_obs = []
        batch_wids = []

        # Wait for at least one request
        msg = request_queue.get()
        if len(msg) == 3:
            workers_done += 1
            all_results.extend(msg[2])
            continue
        wid, obs = msg
        batch_obs.append(obs)
        batch_wids.append(wid)

        # Collect more (non-blocking, with short timeout)
        deadline = time.time() + args.batch_timeout
        while len(batch_obs) < n_workers:
            try:
                remaining = max(0.0005, deadline - time.time())
                msg = request_queue.get(timeout=remaining)
                if len(msg) == 3:
                    workers_done += 1
                    all_results.extend(msg[2])
                    continue
                wid, obs = msg
                batch_obs.append(obs)
                batch_wids.append(wid)
            except:
                break
            if time.time() >= deadline:
                break

        if not batch_obs:
            continue

        # Batched inference
        action_lists = batched_infer(batch_obs)
        n_infer_calls += len(batch_obs)
        n_batches += 1

        for wid, actions in zip(batch_wids, action_lists):
            response_queues[wid].put(actions)

    eval_time = time.time() - eval_start

    for w in workers:
        w.join(timeout=10)

    # ---- Report ----
    task_ok = collections.defaultdict(int)
    task_n = collections.defaultdict(int)
    for tid, success in all_results:
        task_n[tid] += 1
        if success:
            task_ok[tid] += 1

    total_ok = total_n = 0
    for tid in range(suite.n_tasks):
        ok, n = task_ok[tid], task_n[tid]
        total_ok += ok
        total_n += n
        task = suite.get_task(tid)
        print(f"  Task {tid} [{task.language}]: {ok}/{n} ({ok/n*100:.0f}%)")

    print(f"Success: {total_ok}/{total_n} ({total_ok/total_n*100:.1f}%)")
    print(f"Eval time: {eval_time:.1f}s ({eval_time/total_n:.2f}s/episode)")
    print(f"Batching: {n_infer_calls} calls in {n_batches} batches (avg {n_infer_calls/max(1,n_batches):.1f}/batch)")
    print(f"Model load: {load_time:.1f}s (excluded)")
    print(f"RESULT|{args.task_suite_name}|{total_ok}|{total_n}|{eval_time:.1f}")


if __name__ == "__main__":
    main()
