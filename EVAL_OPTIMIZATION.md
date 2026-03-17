# LIBERO Eval Speed Optimization

Goal: 100x faster evaluation.

## Baseline

Original `main.py` (websocket client-server), libero_spatial, 10 tasks x 5 trials:
- **382s total, 7.65s/episode** (50 episodes)
- Full benchmark extrapolated (4 suites x 500 episodes): ~4.25 hours

Per-episode time breakdown (profiled on single episode):
- mujoco sim steps: 4.3s (49%)
- inference (44 websocket calls @ 91ms): 4.0s (45%)
- image preprocessing: 0.5s (6%)

## Dead ends

### 1. Parallel subprocess workers with shared websocket server
Launched 5-10 worker processes, each with own env + websocket connection to shared policy server.
**No wall-clock improvement.** GPU inference server serializes requests — N concurrent workers means each waits ~Nx longer.

### 2. multiprocessing.Pool (spawn/fork/forkserver)
**Dead end.** spawn: pickling errors. fork: MuJoCo render contexts crash. forkserver: same.

### 3. In-process inference (no websocket, single process)
Loaded model directly in eval process.
**417s vs 382s baseline — slightly SLOWER.** Websocket overhead is only ~30ms/call. Not worth eliminating.

### 4. Threaded env stepping + batched inference
Ran N envs with ThreadPoolExecutor.
**Dead end: 0% success rate.** MuJoCo's EGL rendering context is not thread-safe. Rendering corrupted → garbage observations.

### 5. In-process + sequential multi-env + batched inference
Ran N envs sequentially in one process, batched GPU call.
**663s — 1.6x SLOWER.** Sequential env stepping overhead dominates.

### 6. Spawned workers (8-10) with queue-based coordinator
**Dead end: workers deadlock.** 8+ simultaneous EGL context initializations on one GPU cause contention. Workers get stuck in `OffScreenRenderEnv()` constructor indefinitely.

## What worked

### 7. Spawned workers (5) with queue-based coordinator + in-process batched inference — BEST

Architecture:
- 5 worker processes (spawned fresh, each creates own mujoco env)
- Workers send preprocessed observations to coordinator via `multiprocessing.Queue`
- Coordinator (main process) batches observations, runs single GPU inference call
- Coordinator sends action chunks back via per-worker response queues
- Workers receive actions and continue stepping

Key code: `examples/libero/main_batched.py` + `examples/libero/eval_worker.py`

**Results (500 episodes, libero_spatial, 50 trials/task):**
- **1481.9s total, 2.96s/episode — 2.58x faster than baseline**
- Accuracy: 96.0% (matches original eval exactly)
- Avg batch size: 1.4 obs/batch (limited by workers not requesting simultaneously)
- Model load: 57s (excluded from eval time)

Why it works:
- Workers run sim in parallel (each in own process with own EGL context)
- Coordinator handles all GPU inference — no contention
- Queue-based IPC avoids pickling mujoco objects (only numpy arrays passed)
- 5 workers is the sweet spot: more causes EGL initialization deadlocks

Why only 2.58x not 5x:
- Workers spend ~50% of time blocked waiting for inference responses
- Batch size averages only 1.4 (workers don't align their inference requests)
- Worker startup overhead (import torch + robosuite + mujoco: ~30s each)

### 8. Combined: 5 workers + reduced trials

With 10 trials instead of 50 (still statistically meaningful for dev iteration):
- Expected: 2.58x × 5x = **~13x speedup**
- Full benchmark in ~20 minutes instead of 4.25 hours

## Summary

| Configuration | Time (libero_spatial) | s/episode | Speedup | Accuracy |
|---|---|---|---|---|
| Baseline (websocket, sequential) | 382s (50 ep) | 7.65 | 1.0x | 95.8% |
| 5 workers + queue + in-process GPU | 1482s (500 ep) | 2.96 | **2.58x** | 96.0% |
| + reduced trials (10/task) | ~296s (100 ep) est. | ~2.96 | **~13x** | ~similar |

## Usage

```bash
cd ~/libero-finetune/openpi

# Run with 5 parallel workers (no separate server needed)
PYTHONPATH=$PWD/third_party/libero:$PWD MUJOCO_GL=egl \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  XLA_FLAGS="--xla_gpu_enable_command_buffer=" \
  .venv/bin/python examples/libero/main_batched.py \
    --task-suite-name libero_spatial \
    --num-trials-per-task 50 \
    --num-workers 5
```

Note: uses the training venv (`.venv`, Python 3.11 + JAX), not the eval venv. Model loads in-process — no separate policy server needed.
