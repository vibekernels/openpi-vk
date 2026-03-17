"""Compute normalization statistics for a training config.

Optimized parallel pipeline for LeRobot datasets (LIBERO, ALOHA, DROID, etc.):
  - Reads parquet files directly with pyarrow, skipping image columns
  - Workers are pure numpy/pyarrow (no JAX/torch imports)
  - Workers return only compact stats (~100 KB each), not raw arrays
  - Uses forkserver to avoid inheriting parent's heavy import memory
  - Reservoir sampling for quantile computation
"""

import gc
import time

import numpy as np


# ============================================================
# Worker function — pure numpy/pyarrow, no heavy imports.
# Defined at module top level so it's picklable for forkserver.
# ============================================================

def process_lerobot_episode(args):
    """Process a single LeRobot parquet file and return compact statistics.

    Reads state/actions columns directly via pyarrow, skipping images entirely.
    Returns ~100 KB of aggregated stats per episode.
    Only imports numpy and pyarrow — no torch/JAX/lerobot.
    """
    import pyarrow.parquet as pq

    (episode_file, state_col, actions_col, action_horizon,
     delta_mask_arr, sample_fraction, reservoir_per_episode) = args

    try:
        table = pq.read_table(episode_file, columns=[state_col, actions_col])

        n = len(table)
        if n == 0:
            return None

        sc = table.column(state_col).combine_chunks()
        states = sc.values.to_numpy(zero_copy_only=False).reshape(n, -1)
        ac = table.column(actions_col).combine_chunks()
        actions = ac.values.to_numpy(zero_copy_only=False).reshape(n, -1)
        del table, sc, ac

        # Apply delta transform if configured
        if delta_mask_arr is not None:
            dims = delta_mask_arr.shape[-1]
            actions = actions.copy()
            actions[..., :dims] = np.where(
                delta_mask_arr,
                actions[..., :dims] - states[..., :dims],
                actions[..., :dims]
            )

        result = {
            "state_count": n,
            "state_sum": np.sum(states, axis=0),
            "state_sum_sq": np.sum(states**2, axis=0),
            "state_min": np.min(states, axis=0),
            "state_max": np.max(states, axis=0),
            "action_count": n,
            "action_sum": np.sum(actions, axis=0),
            "action_sum_sq": np.sum(actions**2, axis=0),
            "action_min": np.min(actions, axis=0),
            "action_max": np.max(actions, axis=0),
        }

        # Build action chunks for reservoir (quantile computation)
        num_chunks = n - action_horizon + 1
        if num_chunks <= 0:
            return result

        rng = np.random.RandomState(hash(str(episode_file)) % (2**31))
        n_chunk_samples = max(1, int(num_chunks * sample_fraction))

        if sample_fraction < 1.0 and n_chunk_samples < num_chunks:
            chunk_indices = rng.choice(num_chunks, size=n_chunk_samples, replace=False)
        else:
            chunk_indices = np.arange(num_chunks)

        window_offsets = np.arange(action_horizon)
        gather_indices = chunk_indices[:, None] + window_offsets[None, :]
        action_windows = actions[gather_indices]  # (n_samples, H, D)

        n_samples = action_windows.shape[0]

        n_reservoir = min(reservoir_per_episode, n_samples)
        if n_reservoir < n_samples:
            res_idx = rng.choice(n_samples, size=n_reservoir, replace=False)
            result["reservoir"] = action_windows[res_idx]
        else:
            result["reservoir"] = action_windows

        return result

    except Exception as e:
        print(f"Error processing {episode_file}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# Main
# ============================================================

def main(
    config_name: str,
    max_episodes: int | None = None,
    num_workers: int = 24,
    sample_fraction: float = 0.1,
    reservoir_per_episode: int = 200,
    max_reservoir: int = 500_000,
):
    """Compute normalization statistics for a training config.

    Reads LeRobot parquet files directly with pyarrow in parallel workers,
    skipping image columns entirely. Uses streaming accumulation + reservoir
    sampling for memory-efficient quantile computation.

    Args:
        config_name: Name of the training config (e.g. pi05_libero, pi0_libero)
        max_episodes: Maximum episodes to process (None = all)
        num_workers: Number of parallel workers
        sample_fraction: Fraction of action chunks to sample per episode (0.1 = 10%)
        reservoir_per_episode: Chunk samples kept per episode for quantile computation
        max_reservoir: Maximum total reservoir size for quantile computation
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pathlib import Path

    import pyarrow.parquet as pq

    import openpi.shared.normalize as normalize
    import openpi.training.config as _config
    import openpi.transforms as transforms

    t_start = time.time()

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Data config must have a repo_id")

    # Resolve dataset root via LeRobot metadata
    from lerobot.common.datasets import lerobot_dataset
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    data_root = Path(meta.root)

    # Find all episode parquet files
    episode_files = sorted(data_root.glob("data/chunk-*/episode_*.parquet"))
    print(f"Found {len(episode_files)} episode files in {data_root}")

    if max_episodes is not None:
        episode_files = episode_files[:max_episodes]
        print(f"Limited to {len(episode_files)} files")

    if not episode_files:
        raise ValueError(f"No episode parquet files found in {data_root}/data/chunk-*/")

    # Detect delta transform mask from config
    delta_mask = None
    for transform in data_config.data_transforms.inputs:
        if isinstance(transform, transforms.DeltaActions):
            delta_mask = transform.mask
            break
    delta_mask_arr = np.asarray(delta_mask) if delta_mask is not None else None
    if delta_mask_arr is not None:
        print(f"Using delta transform with mask: {delta_mask}")

    action_horizon = config.model.action_horizon
    action_dim = config.model.action_dim

    # LeRobot parquet column names
    state_col = "state"
    actions_col = "actions"

    num_workers = max(1, min(num_workers, len(episode_files)))
    print(f"Using {num_workers} workers (forkserver)")

    # Detect raw action dim from first file
    sample_table = pq.read_table(str(episode_files[0]), columns=[actions_col])
    sample_col = sample_table.column(actions_col).combine_chunks()
    action_dim_raw = sample_col.values.to_numpy(zero_copy_only=False).reshape(len(sample_table), -1).shape[1]
    del sample_table, sample_col

    # ====== Streaming accumulators ======
    total_state_count = 0
    state_sum = state_sum_sq = state_min = state_max = None

    total_action_count = 0
    action_sum = action_sum_sq = action_min = action_max = None

    reservoir_arr = np.empty((max_reservoir, action_horizon, action_dim_raw), dtype=np.float64)
    reservoir_filled = 0
    reservoir_seen = 0
    rng_reservoir = np.random.RandomState(42)

    num_processed = 0
    t_processing_start = time.time()

    ctx = mp.get_context("forkserver")

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        future_to_file = {
            executor.submit(
                process_lerobot_episode,
                (str(ep), state_col, actions_col, action_horizon,
                 delta_mask_arr, sample_fraction, reservoir_per_episode)
            ): ep
            for ep in episode_files
        }

        for future in as_completed(future_to_file):
            try:
                result = future.result()
            except Exception as e:
                ep = future_to_file[future]
                print(f"Error processing {ep}: {e}")
                continue

            if result is None:
                continue

            # --- State stats ---
            total_state_count += result["state_count"]
            if state_sum is None:
                state_sum = result["state_sum"].copy()
                state_sum_sq = result["state_sum_sq"].copy()
                state_min = result["state_min"].copy()
                state_max = result["state_max"].copy()
            else:
                state_sum += result["state_sum"]
                state_sum_sq += result["state_sum_sq"]
                np.minimum(state_min, result["state_min"], out=state_min)
                np.maximum(state_max, result["state_max"], out=state_max)

            # --- Action stats ---
            total_action_count += result["action_count"]
            if action_sum is None:
                action_sum = result["action_sum"].copy()
                action_sum_sq = result["action_sum_sq"].copy()
                action_min = result["action_min"].copy()
                action_max = result["action_max"].copy()
            else:
                action_sum += result["action_sum"]
                action_sum_sq += result["action_sum_sq"]
                np.minimum(action_min, result["action_min"], out=action_min)
                np.maximum(action_max, result["action_max"], out=action_max)

            # --- Reservoir sampling ---
            if "reservoir" in result:
                ep_res = result["reservoir"]
                n_new = ep_res.shape[0]

                if reservoir_filled < max_reservoir:
                    space = max_reservoir - reservoir_filled
                    n_direct = min(n_new, space)
                    reservoir_arr[reservoir_filled:reservoir_filled + n_direct] = ep_res[:n_direct]
                    reservoir_filled += n_direct
                    reservoir_seen += n_direct
                    ep_res = ep_res[n_direct:]
                    n_new = ep_res.shape[0]

                for i in range(n_new):
                    reservoir_seen += 1
                    j = rng_reservoir.randint(0, reservoir_seen)
                    if j < max_reservoir:
                        reservoir_arr[j] = ep_res[i]

            num_processed += 1
            if num_processed % 200 == 0:
                elapsed = time.time() - t_processing_start
                eps_per_sec = num_processed / elapsed
                remaining = (len(episode_files) - num_processed) / eps_per_sec if eps_per_sec > 0 else 0
                print(f"Processed {num_processed}/{len(episode_files)} episodes "
                      f"({eps_per_sec:.1f} eps/sec, ~{remaining:.0f}s remaining, "
                      f"reservoir: {reservoir_filled}/{max_reservoir})")

    t_processing_end = time.time()
    print(f"\nProcessed {num_processed} episodes in {t_processing_end - t_processing_start:.1f}s "
          f"({num_processed / (t_processing_end - t_processing_start):.1f} eps/sec)")

    if num_processed == 0:
        raise RuntimeError("No episodes processed")

    # Trim reservoir
    reservoir_arr = reservoir_arr[:reservoir_filled]
    print(f"Reservoir: {reservoir_filled:,} samples ({reservoir_arr.nbytes / 1024**2:.0f} MB)")

    gc.collect()

    # ====== Compute final statistics ======
    print("\nComputing final statistics...")

    state_mean = state_sum / total_state_count
    state_var = (state_sum_sq / total_state_count) - state_mean**2
    state_std = np.sqrt(np.maximum(0, state_var))

    action_mean = action_sum / total_action_count
    action_var = (action_sum_sq / total_action_count) - action_mean**2
    action_std = np.sqrt(np.maximum(0, action_var))

    # Quantiles from reservoir
    if reservoir_filled > 0:
        print(f"Computing quantiles from {reservoir_filled:,} reservoir samples...")
        res_flat = reservoir_arr.reshape(-1, reservoir_arr.shape[-1])
        action_q01 = np.percentile(res_flat, 1, axis=0)
        action_q99 = np.percentile(res_flat, 99, axis=0)
        print(f"  q01/q99 from {len(res_flat):,} data points")
    else:
        action_q01 = action_min
        action_q99 = action_max

    del reservoir_arr
    gc.collect()

    # ====== Build NormStats and save ======
    norm_stats = {
        "state": normalize.NormStats(
            mean=transforms.pad_to_dim(state_mean, action_dim),
            std=transforms.pad_to_dim(state_std, action_dim),
            q01=transforms.pad_to_dim(state_min, action_dim),
            q99=transforms.pad_to_dim(state_max, action_dim),
        ),
        "actions": normalize.NormStats(
            mean=transforms.pad_to_dim(action_mean, action_dim),
            std=transforms.pad_to_dim(action_std, action_dim),
            q01=transforms.pad_to_dim(action_q01, action_dim),
            q99=transforms.pad_to_dim(action_q99, action_dim),
        ),
    }

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)

    # Summary
    t_end = time.time()
    print(f"\nState  mean: {state_mean}")
    print(f"Action mean: {action_mean}")
    print(f"State  std:  {state_std}")
    print(f"Action std:  {action_std}")
    print(f"Action q01:  {action_q01}")
    print(f"Action q99:  {action_q99}")

    print(f"\n{'='*60}")
    print(f"Total time: {t_end - t_start:.1f}s")
    print(f"Processing: {t_processing_end - t_processing_start:.1f}s "
          f"({num_processed / (t_processing_end - t_processing_start):.1f} eps/sec)")
    print(f"{'='*60}")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
