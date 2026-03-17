# Norm Stats Computation Optimization

## Baseline
- **Script**: `scripts/compute_norm_stats.py`
- **Dataset**: 10,000 episodes, ~17K rows each, 128 CPUs, 512 GB RAM
- **Baseline speed**: ~3.3 eps/sec (32 workers), ~50 min estimated for 10K episodes
- **Baseline problem**: OOM — accumulated all raw chunk data (~110 GB) in memory before aggregation

## Key Findings from Prior Session
- More workers helped (32 → 64 showed improvement) for the old code
- PyArrow column selection didn't work because `observation.state` and `action` columns contain nested arrays (object dtype)
- The original code already had: vectorized numpy (no iterrows), column selection, sampled chunks (10%)

## Root Causes of OOM
1. **Worker results too large**: Each worker returned a (960, 960) float64 outer product matrix (~7.4 MB) for streaming correlation. With 10K episodes, IPC serialization alone was ~74 GB.
2. **Fork inherited heavy imports**: Workers forked from main process inherited JAX/torch/omnigibson memory. With 64 workers, this multiplied the ~4 GB import footprint.
3. **Reservoir as Python list**: Growing list of 1M numpy arrays had significant Python object overhead.

## Optimizations Applied

### Round 1: Streaming Aggregation (eliminated ~110 GB accumulation)
- Workers return compact stats (count/sum/sum_sq/min/max) instead of raw chunk arrays
- Main process accumulates incrementally — O(1) memory regardless of episode count
- Reservoir sampling for quantile computation instead of collecting all data
- **Result**: 100 eps in 4.6s (21.8 eps/sec), up from 30s. But still OOMed at 10K.

### Round 2: Memory-Safe Design (eliminated remaining OOM sources)
- **Removed outer product from workers**: Correlation computed from reservoir at the end, not streamed. Worker results dropped from ~8.5 MB to ~100 KB each.
- **Inlined `extract_state_from_proprio`**: Workers are pure numpy/pyarrow — no JAX/torch/omnigibson imports. PROPRIOCEPTION_INDICES extracted as plain dict of ints in main, passed to workers.
- **`forkserver` start method**: Workers fork from a clean process, not the bloated main process with heavy imports.
- **Pre-allocated float64 reservoir**: Single numpy array (500K × 30 × 23 × 8B = 2.76 GB max), no Python list growth.
- **Result**: 10K episodes in 161s (62.1 eps/sec), peak memory ~5 GB. No OOM.

### Round 3: Read path + worker tuning
- **PyArrow direct read**: Replaced `pd.read_parquet` + `np.stack` with `pq.read_table` + `combine_chunks().values.to_numpy().reshape()`. Avoids pandas DataFrame overhead. ~1.2x faster on hot cache reads.
- **Batch reservoir insertion**: Bulk `memcpy` while filling, per-item replacement only after full.
- **Worker count tuning**: forkserver startup overhead means fewer workers is better. Tested 16/24/32/48/64/96/128 workers. Sweet spot: **24 workers** (91.6 eps/sec) > 32 (88.5) > 64 (63.4) > 96 (31.4).
- **Result**: 10K episodes in **109s** (91.6 eps/sec), peak memory ~5 GB.

### Correctness Verification
- Mean/std/per-timestamp stats: **exact match** (diff < 1e-9) between original and optimized
- Cholesky correlation: **exact match** (diff = 0.0) when using same reservoir parameters
- Differences only arise from different reservoir sample counts, not computation errors

## Results

| Optimization | Workers | Episodes/sec | Total time (10K) | Peak memory | Kept? |
|---|---|---|---|---|---|
| Baseline (original) | 32 | ~3.3 | ~50 min (est.) | OOM | - |
| Round 1: streaming stats | 32 | 21.8 | OOM at scale | ~110 GB+ | Replaced |
| Round 2: memory-safe | 64 | 62.1 | 2.7 min | ~5 GB | Replaced |
| **Round 3: pyarrow + tuning** | **24** | **91.6** | **1.8 min** | **~5 GB** | **Yes** |

**Overall: ~28x throughput improvement**, from ~3.3 eps/sec to 91.6 eps/sec.

## Architecture (Current)

```
Main process (heavy imports: JAX, torch, omnigibson, openpi)
  │
  ├── Extract config values, delta_mask, proprio_slices as plain Python types
  │
  └── forkserver ProcessPoolExecutor (24 workers)
        │
        Workers (only import: numpy, pyarrow)
          ├── Read parquet via pyarrow (2 columns, flatten+reshape)
          ├── Inlined state extraction (numpy slicing)
          ├── Vectorized delta transform
          ├── Compute compact stats (count/sum/sum_sq/min/max)
          ├── Sample reservoir chunks (200 per episode)
          └── Return ~100 KB result dict
        │
  Main loop: accumulate stats + reservoir sampling (500K cap)
  │
  Post-processing:
    ├── Quantiles from reservoir (np.percentile)
    ├── Correlation from reservoir (X.T @ X on 500K×960 matrix)
    ├── Cholesky decomposition
    └── Save norm_stats.json
```
