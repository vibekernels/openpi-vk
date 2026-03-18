# pi0.5 LIBERO Finetuning Optimizations

## Setup
- **Hardware**: 1x H100 80GB SXM
- **Model**: pi0.5 (full finetune, float32 params, bfloat16 compute)
- **Dataset**: physical-intelligence/libero (1158 episodes)
- **Framework**: JAX/Flax (openpi)
- **Metric**: Average training loss at 5 minutes of actual training (excluding JIT compilation / init)

### Memory constraints
The original config uses batch_size=256 + EMA, which requires ~150GB and OOMs on a single H100.
All experiments use **batch_size=16, no EMA** to fit in 80GB.

## Baseline

| Config | batch_size | warmup | peak_lr | decay_lr | EMA | Steps in 5min | 5min Loss |
|--------|-----------|--------|---------|----------|-----|---------------|-----------|
| `pi05_libero_baseline` | 16 | 10,000 | 5e-5 | 5e-5 (constant) | None | ~375 | **0.041** |

**Key observation**: With 10,000-step warmup and only ~375 steps in 5 minutes, the effective LR never exceeds `5e-5 * (375/10000) ≈ 1.9e-7`. The model barely learns.

### Baseline loss curve
| Step | Time | Loss | Grad Norm |
|------|------|------|-----------|
| 0 | 0:00 | 0.0918 | 0.9488 |
| 50 | 0:40 | 0.0818 | 0.8952 |
| 100 | 1:21 | 0.0686 | 0.6748 |
| 150 | 2:01 | 0.0636 | 0.6398 |
| 200 | 2:42 | 0.0561 | 0.6008 |
| 250 | 3:22 | 0.0499 | 0.5405 |
| 300 | 4:02 | 0.0468 | 0.5419 |
| 350 | 4:42 | 0.0422 | 0.5099 |
| 400 | 5:22 | 0.0404 | 0.5721 |

**Target**: 5min loss ≤ 0.0205 (half of baseline)

## Experiments

### Experiment 1: Short warmup + higher LR
**Hypothesis**: Reducing warmup from 10k to 50 steps and increasing peak_lr to 2.5e-4 (5x) lets the model reach full learning rate almost immediately, dramatically accelerating convergence.

| Config | batch_size | warmup | peak_lr | decay_lr | EMA |
|--------|-----------|--------|---------|----------|-----|
| `pi05_libero_exp1` | 16 | 50 | 2.5e-4 | 1e-6 | None |

**Result**: _pending_
