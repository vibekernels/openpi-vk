"""Compute normalization statistics for BEHAVIOR-1K dataset.

Computes mean, std, min, max from dataset in parallel.
Supports per-timestamp normalization and action correlation matrices.

Memory-safe design:
  - Workers are pure numpy/pandas (no JAX/torch/omnigibson imports)
  - Workers return only compact stats (~100 KB each), not raw arrays
  - Correlation computed from reservoir at the end, not streamed outer products
  - Uses forkserver to avoid inheriting parent's heavy import memory
"""

import numpy as np
import time
import gc


# ============================================================
# Worker function — pure numpy/pandas, no heavy imports.
# Defined at module top level so it's picklable for forkserver.
# ============================================================

def _extract_state_inlined(proprio_data, proprio_slices):
    """Inline version of extract_state_from_proprio using pre-extracted slice indices."""
    base_qvel = proprio_data[..., proprio_slices["base_qvel_s"]:proprio_slices["base_qvel_e"]]
    trunk_qpos = proprio_data[..., proprio_slices["trunk_qpos_s"]:proprio_slices["trunk_qpos_e"]]
    arm_left_qpos = proprio_data[..., proprio_slices["arm_left_qpos_s"]:proprio_slices["arm_left_qpos_e"]]
    arm_right_qpos = proprio_data[..., proprio_slices["arm_right_qpos_s"]:proprio_slices["arm_right_qpos_e"]]
    left_gripper_raw = proprio_data[..., proprio_slices["gripper_left_qpos_s"]:proprio_slices["gripper_left_qpos_e"]].sum(axis=-1, keepdims=True)
    right_gripper_raw = proprio_data[..., proprio_slices["gripper_right_qpos_s"]:proprio_slices["gripper_right_qpos_e"]].sum(axis=-1, keepdims=True)

    MAX_GRIPPER_WIDTH = 0.1
    left_gripper_width = 2.0 * (left_gripper_raw / MAX_GRIPPER_WIDTH) - 1.0
    right_gripper_width = 2.0 * (right_gripper_raw / MAX_GRIPPER_WIDTH) - 1.0

    return np.concatenate([
        base_qvel, trunk_qpos, arm_left_qpos, left_gripper_width,
        arm_right_qpos, right_gripper_width,
    ], axis=-1)


def _apply_delta_batch(states, actions, mask_arr):
    """Vectorized delta transform for batched states/actions."""
    if mask_arr is None:
        return actions
    dims = mask_arr.shape[-1]
    delta = actions.copy()
    delta[..., :dims] = np.where(mask_arr, actions[..., :dims] - states[:, None, :dims], actions[..., :dims])
    return delta


def process_episode_file(args):
    """Process a single episode file and return compact statistics.

    Returns ~100 KB of aggregated stats per episode (no raw chunk arrays).
    Only imports numpy and pandas — no JAX/torch/omnigibson.
    """
    import pyarrow.parquet as pq  # Import inside worker to keep module-level clean

    (episode_file, delta_mask_arr, action_horizon,
     compute_per_timestamp, compute_correlation, sample_fraction,
     reservoir_per_episode, proprio_slices) = args

    try:
        table = pq.read_table(episode_file, columns=["observation.state", "action"])

        n = len(table)
        if n == 0:
            return None

        # Direct pyarrow flatten+reshape — avoids pandas overhead and np.stack
        sc = table.column("observation.state").combine_chunks()
        raw_states = sc.values.to_numpy(zero_copy_only=False).reshape(n, -1)
        ac = table.column("action").combine_chunks()
        raw_actions = ac.values.to_numpy(zero_copy_only=False).reshape(n, -1)
        del table, sc, ac

        # Inlined state extraction (no omnigibson import needed)
        states = _extract_state_inlined(raw_states, proprio_slices)  # (N, 23)
        del raw_states

        # Per-step delta transform
        if delta_mask_arr is not None:
            dims = delta_mask_arr.shape[-1]
            actions = raw_actions.copy()
            actions[..., :dims] = np.where(
                delta_mask_arr,
                raw_actions[..., :dims] - states[..., :dims],
                raw_actions[..., :dims]
            )
        else:
            actions = raw_actions

        # Compact episode-level stats
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

        # Build action chunks if needed
        if not (compute_per_timestamp or compute_correlation):
            return result

        num_chunks = n - action_horizon + 1
        if num_chunks <= 0:
            return result

        # Sample chunk indices
        rng = np.random.RandomState(hash(str(episode_file)) % (2**31))
        n_chunk_samples = max(1, int(num_chunks * sample_fraction))

        if sample_fraction < 1.0 and n_chunk_samples < num_chunks:
            chunk_indices = rng.choice(num_chunks, size=n_chunk_samples, replace=False)
        else:
            chunk_indices = np.arange(num_chunks)

        # Build sampled chunks using vectorized stride indexing
        window_offsets = np.arange(action_horizon)
        gather_indices = chunk_indices[:, None] + window_offsets[None, :]
        action_windows = raw_actions[gather_indices]  # (n_samples, H, D)
        chunk_states = states[chunk_indices]

        sampled_chunks = _apply_delta_batch(chunk_states, action_windows, delta_mask_arr)
        n_samples = sampled_chunks.shape[0]

        # Per-timestamp compact stats (tiny: H×D arrays)
        if compute_per_timestamp:
            result["pts_count"] = n_samples
            result["pts_sum"] = np.sum(sampled_chunks, axis=0)
            result["pts_sum_sq"] = np.sum(sampled_chunks**2, axis=0)
            result["pts_min"] = np.min(sampled_chunks, axis=0)
            result["pts_max"] = np.max(sampled_chunks, axis=0)

        # Reservoir sample for quantile + correlation computation
        n_reservoir = min(reservoir_per_episode, n_samples)
        if n_reservoir < n_samples:
            res_idx = rng.choice(n_samples, size=n_reservoir, replace=False)
            result["reservoir"] = sampled_chunks[res_idx]
        else:
            result["reservoir"] = sampled_chunks

        return result

    except Exception as e:
        print(f"Error processing {episode_file}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# Correlation computation from reservoir (main process only)
# ============================================================

def compute_correlation_from_reservoir(reservoir_arr, action_horizon, target_action_dim,
                                        max_correlation_samples=2_000_000):
    """Compute correlation matrix, Cholesky, spatial and temporal averages from reservoir samples."""
    action_dim = reservoir_arr.shape[-1]  # 23
    n_total = reservoir_arr.shape[0]

    # Pad to target_action_dim
    if action_dim < target_action_dim:
        padding = np.zeros((n_total, action_horizon, target_action_dim - action_dim), dtype=reservoir_arr.dtype)
        padded = np.concatenate([reservoir_arr, padding], axis=2)
    else:
        padded = reservoir_arr[:, :, :target_action_dim]

    F = action_horizon * target_action_dim
    flat = padded.reshape(n_total, F).astype(np.float64)
    del padded

    # Subsample if too many
    if n_total > max_correlation_samples:
        print(f"Subsampling {max_correlation_samples:,} from {n_total:,} for correlation")
        rng = np.random.RandomState(42)
        idx = rng.choice(n_total, size=max_correlation_samples, replace=False)
        flat = flat[idx]
        print(f"Using {flat.shape[0]:,} samples")
    else:
        print(f"Using all {n_total:,} samples")

    # Compute mean, std, covariance
    chunk_mean = np.mean(flat, axis=0)
    chunk_std = np.std(flat, axis=0)

    constant_dims = chunk_std < 1e-6
    print(f"Found {np.sum(constant_dims)} constant/padded dimensions (will be set to identity)")

    # Normalize non-constant dims
    normalized = flat.copy()
    normalized[:, ~constant_dims] = (flat[:, ~constant_dims] - chunk_mean[~constant_dims]) / chunk_std[~constant_dims]
    del flat

    # Compute covariance = correlation (since data is normalized)
    cov_matrix = (normalized.T @ normalized) / normalized.shape[0]
    # Correct for mean (should be ~0 after normalization, but be precise)
    norm_mean = np.mean(normalized, axis=0)
    cov_matrix -= np.outer(norm_mean, norm_mean)

    # Clean up constant dims
    cov_matrix[constant_dims, :] = 0.0
    cov_matrix[:, constant_dims] = 0.0
    np.fill_diagonal(cov_matrix, 1.0)

    print(f"Correlation matrix shape: {cov_matrix.shape}")
    print(f"Diagonal: min={np.min(np.diag(cov_matrix)):.6f}, max={np.max(np.diag(cov_matrix)):.6f}")
    print(f"Matrix check: has NaN={np.any(np.isnan(cov_matrix))}, has Inf={np.any(np.isinf(cov_matrix))}")

    # Regularization
    epsilon = 1e-6
    cov_matrix_reg = cov_matrix + epsilon * np.eye(F)

    eigenvalues = np.linalg.eigvalsh(cov_matrix_reg)
    min_eigenvalue = np.min(eigenvalues)
    print(f"Min eigenvalue after regularization: {min_eigenvalue:.6e}")

    if min_eigenvalue <= 0:
        print(f"WARNING: Not positive definite! Adding stronger regularization...")
        epsilon = max(1e-5, -min_eigenvalue + 1e-5)
        cov_matrix_reg = cov_matrix + epsilon * np.eye(F)
        eigenvalues = np.linalg.eigvalsh(cov_matrix_reg)
        print(f"New min eigenvalue: {np.min(eigenvalues):.6e}")

    # Cholesky
    chol_lower = np.linalg.cholesky(cov_matrix_reg)
    print(f"Cholesky decomposition successful! Shape: {chol_lower.shape}")

    reconstructed = chol_lower @ chol_lower.T
    err = np.linalg.norm(reconstructed - cov_matrix_reg, 'fro') / np.linalg.norm(cov_matrix_reg, 'fro')
    print(f"Cholesky reconstruction error: {err:.6e}")

    # Averaged spatial correlation (dim × dim)
    print("\nComputing averaged spatial correlation (dim × dim)...")
    spatial_corrs = []
    for t in range(action_horizon):
        s, e = t * target_action_dim, (t + 1) * target_action_dim
        spatial_corrs.append(cov_matrix[s:e, s:e])
    avg_spatial_corr = np.mean(spatial_corrs, axis=0)
    np.fill_diagonal(avg_spatial_corr, 1.0)
    print(f"Averaged spatial correlation shape: {avg_spatial_corr.shape}")

    # Averaged temporal correlation (time × time)
    print("Computing averaged temporal correlation (time × time)...")
    dim_is_constant = [np.all(chunk_std[[t * target_action_dim + d for t in range(action_horizon)]] < 1e-6)
                       for d in range(target_action_dim)]
    num_const = sum(dim_is_constant)
    print(f"  Excluding {num_const} constant dimensions from temporal averaging")

    temporal_corrs = []
    for d in range(target_action_dim):
        if dim_is_constant[d]:
            continue
        dim_indices = [t * target_action_dim + d for t in range(action_horizon)]
        temporal_corrs.append(cov_matrix[np.ix_(dim_indices, dim_indices)])

    if temporal_corrs:
        avg_temporal_corr = np.mean(temporal_corrs, axis=0)
        np.fill_diagonal(avg_temporal_corr, 1.0)
        print(f"Averaged temporal correlation shape: {avg_temporal_corr.shape}")
    else:
        avg_temporal_corr = np.eye(action_horizon)

    del normalized, cov_matrix, cov_matrix_reg
    gc.collect()

    return {
        "action_correlation_cholesky": chol_lower,
        "action_correlation_spatial": avg_spatial_corr,
        "action_correlation_temporal": avg_temporal_corr,
    }


# ============================================================
# Main — only place that imports heavy modules
# ============================================================

def main(
    config_name: str,
    max_episodes: int | None = None,
    num_workers: int = 24,
    per_timestamp: bool = False,
    correlation: bool = True,
    sample_fraction: float = 0.1,
    reservoir_per_episode: int = 200,
    max_reservoir: int = 500_000,
    max_correlation_samples: int = 2_000_000,
):
    """Compute normalization statistics for B1K config using parallel processing.

    Args:
        config_name: Name of the training config
        max_episodes: Maximum number of episodes to process (None = all)
        num_workers: Number of parallel workers
        per_timestamp: Whether to compute per-timestamp statistics
        correlation: Whether to compute action correlation matrix
        sample_fraction: Fraction of action chunks to sample per episode (0.1 = 10%).
        reservoir_per_episode: Chunk samples kept per episode for quantile/correlation.
        max_reservoir: Maximum total reservoir size for quantile/correlation computation.
        max_correlation_samples: Max samples for correlation matrix computation.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pathlib import Path
    import tyro

    # Heavy imports — only in main process
    import openpi.transforms as transforms
    from b1k.shared import normalize
    from b1k.training import config as _config
    from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES

    t_start = time.time()

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if not data_config.behavior_dataset_root:
        raise ValueError("This script only works with B1K behavior datasets")

    # Get delta transform mask
    delta_mask = None
    for transform in data_config.data_transforms.inputs:
        if isinstance(transform, transforms.DeltaActions):
            delta_mask = transform.mask
            break
    if delta_mask is None:
        raise ValueError("No DeltaActions transform found in config")
    print(f"Using delta transform with mask: {delta_mask}")
    delta_mask_arr = np.asarray(delta_mask)

    # Extract PROPRIOCEPTION_INDICES as plain dict of ints (serializable, no omnigibson needed in workers)
    idx = PROPRIOCEPTION_INDICES["R1Pro"]
    proprio_slices = {
        "base_qvel_s": idx["base_qvel"].start, "base_qvel_e": idx["base_qvel"].stop,
        "trunk_qpos_s": idx["trunk_qpos"].start, "trunk_qpos_e": idx["trunk_qpos"].stop,
        "arm_left_qpos_s": idx["arm_left_qpos"].start, "arm_left_qpos_e": idx["arm_left_qpos"].stop,
        "arm_right_qpos_s": idx["arm_right_qpos"].start, "arm_right_qpos_e": idx["arm_right_qpos"].stop,
        "gripper_left_qpos_s": idx["gripper_left_qpos"].start, "gripper_left_qpos_e": idx["gripper_left_qpos"].stop,
        "gripper_right_qpos_s": idx["gripper_right_qpos"].start, "gripper_right_qpos_e": idx["gripper_right_qpos"].stop,
    }

    compute_per_timestamp = per_timestamp or data_config.use_per_timestamp_norm
    compute_correlation = correlation
    action_horizon = config.model.action_horizon
    target_action_dim = config.model.action_dim

    if compute_per_timestamp:
        print(f"Computing per-timestamp statistics with action_horizon={action_horizon}")
    if compute_correlation:
        print(f"Computing correlation matrix with action_horizon={action_horizon}")
    if sample_fraction < 1.0:
        print(f"Sampling {sample_fraction*100:.0f}% of action chunks per episode")

    # Find episode files
    data_root = Path(data_config.behavior_dataset_root)
    episode_files = sorted(data_root.glob("data/task-*/episode_*.parquet"))
    print(f"Found {len(episode_files)} episode files")

    if max_episodes is not None:
        episode_files = episode_files[:max_episodes]
        print(f"Limited to {len(episode_files)} files")

    if not episode_files:
        raise ValueError("No episode files found")

    num_workers = max(1, min(num_workers, len(episode_files)))
    print(f"Using {num_workers} workers (forkserver)")

    # ====== Streaming accumulators ======
    total_state_count = 0
    state_sum = state_sum_sq = state_min = state_max = None

    total_action_count = 0
    action_sum = action_sum_sq = action_min = action_max = None

    pts_count = 0
    pts_sum = pts_sum_sq = pts_min = pts_max = None

    # Reservoir: pre-allocated numpy array
    # 500K × (30, 23) × 8 bytes = 2.76 GB
    action_dim_raw = 23  # before padding
    reservoir_arr = np.empty((max_reservoir, action_horizon, action_dim_raw), dtype=np.float64)
    reservoir_filled = 0
    reservoir_seen = 0
    rng_reservoir = np.random.RandomState(42)

    num_processed = 0
    t_processing_start = time.time()

    # Use forkserver to avoid inheriting parent's JAX/torch memory
    ctx = mp.get_context("forkserver")

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        future_to_file = {
            executor.submit(
                process_episode_file,
                (str(ep), delta_mask_arr, action_horizon,
                 compute_per_timestamp, compute_correlation, sample_fraction,
                 reservoir_per_episode, proprio_slices)
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

            # --- Per-timestamp stats ---
            if "pts_count" in result:
                pts_count += result["pts_count"]
                if pts_sum is None:
                    pts_sum = result["pts_sum"].copy()
                    pts_sum_sq = result["pts_sum_sq"].copy()
                    pts_min = result["pts_min"].copy()
                    pts_max = result["pts_max"].copy()
                else:
                    pts_sum += result["pts_sum"]
                    pts_sum_sq += result["pts_sum_sq"]
                    np.minimum(pts_min, result["pts_min"], out=pts_min)
                    np.maximum(pts_max, result["pts_max"], out=pts_max)

            # --- Reservoir sampling (bulk fill, then per-item replacement) ---
            if "reservoir" in result:
                ep_res = result["reservoir"]  # (n, H, D) float32
                n_new = ep_res.shape[0]

                if reservoir_filled < max_reservoir:
                    # Still filling: bulk copy
                    space = max_reservoir - reservoir_filled
                    n_direct = min(n_new, space)
                    reservoir_arr[reservoir_filled:reservoir_filled + n_direct] = ep_res[:n_direct]
                    reservoir_filled += n_direct
                    reservoir_seen += n_direct
                    ep_res = ep_res[n_direct:]  # any remainder goes to replacement
                    n_new = ep_res.shape[0]

                # Replacement sampling for remaining items (most are rejected, so loop is fast)
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

    # Trim reservoir to actual size
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

    final_stats = {
        "state": {"mean": state_mean, "std": state_std, "q01": state_min, "q99": state_max},
        "actions": {"mean": action_mean, "std": action_std, "q01": action_q01, "q99": action_q99},
    }

    # ====== Per-timestamp statistics ======
    per_timestamp_stats = None
    if compute_per_timestamp and pts_count > 0:
        pts_mean = pts_sum / pts_count
        pts_var = (pts_sum_sq / pts_count) - pts_mean**2
        pts_std_val = np.sqrt(np.maximum(0, pts_var))

        if reservoir_filled > 0:
            pts_q01 = np.percentile(reservoir_arr, 1, axis=0)
            pts_q99 = np.percentile(reservoir_arr, 99, axis=0)
        else:
            pts_q01, pts_q99 = pts_min, pts_max

        per_timestamp_stats = {
            "per_timestamp_mean": pts_mean,
            "per_timestamp_std": pts_std_val,
            "per_timestamp_q01": pts_q01,
            "per_timestamp_q99": pts_q99,
        }

    # ====== Correlation matrix (from reservoir, not streamed) ======
    correlation_stats = None
    if compute_correlation:
        if reservoir_filled == 0:
            raise RuntimeError("Correlation requested but reservoir is empty")

        print(f"\nComputing correlation matrix from {reservoir_filled:,} reservoir samples...")
        correlation_stats = compute_correlation_from_reservoir(
            reservoir_arr, action_horizon, target_action_dim, max_correlation_samples
        )

    # Free reservoir
    del reservoir_arr
    gc.collect()

    # ====== Build NormStats and save ======
    correlation_matrix = None
    if compute_correlation:
        if correlation_stats is None:
            raise RuntimeError("Correlation computation failed")
        chol_matrix = correlation_stats["action_correlation_cholesky"]
        expected_dim = action_horizon * target_action_dim
        if chol_matrix.shape[0] != expected_dim:
            raise ValueError(
                f"Correlation matrix size {chol_matrix.shape[0]} != expected {expected_dim}")
        correlation_matrix = chol_matrix

    norm_stats_dict = {}
    for key in ["state", "actions"]:
        stats_kwargs = {}
        for stat_type in ["mean", "std", "q01", "q99"]:
            stat_value = final_stats[key][stat_type]
            stats_kwargs[stat_type] = transforms.pad_to_dim(stat_value, target_action_dim)

        if per_timestamp_stats is not None and key == "actions":
            print("Adding per-timestamp statistics...")
            for stat_type in ["per_timestamp_mean", "per_timestamp_std", "per_timestamp_q01", "per_timestamp_q99"]:
                stat_value = per_timestamp_stats[stat_type]
                padded = np.array([transforms.pad_to_dim(stat_value[t], target_action_dim)
                                   for t in range(stat_value.shape[0])])
                stats_kwargs[stat_type] = padded

        if correlation_matrix is not None and key == "actions":
            stats_kwargs["action_correlation_cholesky"] = correlation_matrix
            if correlation_stats and "action_correlation_spatial" in correlation_stats:
                stats_kwargs["action_correlation_spatial"] = correlation_stats["action_correlation_spatial"]
            if correlation_stats and "action_correlation_temporal" in correlation_stats:
                stats_kwargs["action_correlation_temporal"] = correlation_stats["action_correlation_temporal"]

        norm_stats_dict[key] = normalize.NormStats(**stats_kwargs)

    # Save
    output_path = config.assets_dirs / data_config.repo_id
    output_path.mkdir(parents=True, exist_ok=True)
    normalize.save(output_path, norm_stats_dict)

    # Summary
    t_end = time.time()
    print(f"\n{'='*60}")
    print(f"Total time: {t_end - t_start:.1f}s")
    print(f"Processing: {t_processing_end - t_processing_start:.1f}s "
          f"({num_processed / (t_processing_end - t_processing_start):.1f} eps/sec)")
    print(f"{'='*60}")

    print("\nStatistics Summary (first 23 dims):")
    state_means = np.array(norm_stats_dict["state"].mean[:23])
    action_means = np.array(norm_stats_dict["actions"].mean[:23])
    differences = np.abs(state_means - action_means)
    print("State means:  ", state_means)
    print("Action means: ", action_means)
    print("Max difference:", np.max(differences))

    if per_timestamp_stats is not None:
        per_ts_means = np.array(norm_stats_dict["actions"].per_timestamp_mean)
        print(f"\nPer-timestamp means shape: {per_ts_means.shape}")
        print("First 3 timesteps, first 5 dims:")
        for t in range(min(3, per_ts_means.shape[0])):
            print(f"  t={t}: {per_ts_means[t, :5]}")

    if compute_correlation:
        print(f"\nCorrelation matrices:")
        corr = np.array(norm_stats_dict["actions"].action_correlation_cholesky)
        print(f"  Cholesky: {corr.shape} ({corr.nbytes / 1024**2:.2f} MB)")
        if norm_stats_dict["actions"].action_correlation_spatial is not None:
            s = np.array(norm_stats_dict["actions"].action_correlation_spatial)
            print(f"  Spatial: {s.shape}, d0-d1 corr: {s[0, 1]:.4f}")
        if norm_stats_dict["actions"].action_correlation_temporal is not None:
            t = np.array(norm_stats_dict["actions"].action_correlation_temporal)
            print(f"  Temporal: {t.shape}, t0-t1 corr: {t[0, 1]:.4f}")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
