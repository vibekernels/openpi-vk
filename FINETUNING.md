# PI0.5 LoRA Fine-tuning on LIBERO — Setup Notes

## What's already on /workspace (persists across sessions)

| Path | Contents |
|------|----------|
| `/workspace/openpi_venv` | Python 3.11 venv with JAX 0.5.3 + all openpi deps (~13 GB) |
| `/workspace/huggingface_cache` | HF token, LIBERO parquet files, xet cache (~47 GB) |
| `/workspace/hf_datasets_cache` | Arrow cache built from parquet on first run (~10–15 GB) |
| `/home/ubuntu/.cache/openpi` | pi05_base GCS checkpoint (~20 GB, downloaded on first train run) |

The `openpi/` repo itself is **not** persisted — it must be re-cloned each session (fast, <1 min).
The norm stats file lives inside the repo and must be recomputed if the repo is re-cloned (takes ~40 min).

## Session start checklist

Run these at the start of every new session:

```bash
# 1. Re-clone openpi (if not already present)
cd ~/libero-finetune
[ -d openpi ] || git clone https://github.com/Physical-Intelligence/openpi.git

# 2. Restore symlinks
[ -L openpi/.venv ] || ln -s /workspace/openpi_venv openpi/.venv
[ -L ~/.cache/huggingface ] || ln -s /workspace/huggingface_cache ~/.cache/huggingface

# 3. Set dataset cache env var
export HF_DATASETS_CACHE=/workspace/hf_datasets_cache

# 4. Restore norm stats (avoids ~40 min recompute)
mkdir -p openpi/assets
cp -r /workspace/openpi_assets/* openpi/assets/

# 5. Add pi05_libero_lora config to config.py (see section below)
#    Check first: grep -q pi05_libero_lora openpi/src/openpi/training/config.py && echo "already present"

cd openpi
```

## The pi05_libero_lora config

After cloning, two changes are needed:

### 1. Increase LoRA alpha in `openpi/src/openpi/models/gemma.py`

Find the `gemma_2b_lora` and `gemma_300m_lora` variant blocks and change alpha to 4x:

```python
# gemma_2b_lora: change alpha from 16.0 to 64.0
lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=64.0), "ffn": lora.LoRAConfig(rank=16, alpha=64.0)},

# gemma_300m_lora: change alpha from 32.0 to 128.0
lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=128.0), "ffn": lora.LoRAConfig(rank=32, alpha=128.0)},
```

### 2. Add config to `openpi/src/openpi/training/config.py`

Add this block immediately after the `pi05_libero` config (around line 764). Also add `import openpi.shared.nnx_utils as nnx_utils` to the imports at the top of the file.

```python
TrainConfig(
    name="pi05_libero_lora",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",       # LoRA rank-16, alpha=64 (4x)
        action_expert_variant="gemma_300m_lora",  # LoRA rank-32, alpha=128 (4x)
    ),
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,  # pi0.5 uses False (unlike pi0)
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    batch_size=24,            # RTX 4090 24 GB; OOMs at 32
    num_train_steps=5_000,    # LoRA converges by ~5k with optimized hyperparams
    log_interval=100,
    save_interval=1_000,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=20,
        peak_lr=5e-5,
        decay_steps=5_000,    # match training length for proper annealing
        decay_lr=1e-6,
    ),
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),    # freezes all LLM weights except LoRA params
    ema_decay=None,           # disabled for LoRA fine-tuning
),
```

Verify it loaded correctly:
```bash
.venv/bin/python -c "from openpi.training.config import get_config; print(get_config('pi05_libero_lora').name)"
# should print: pi05_libero_lora
```

## Training commands

Run from `~/libero-finetune/openpi/`. Use the venv python directly — `uv` is not installed in these containers.

```bash
cd ~/libero-finetune/openpi
export HF_DATASETS_CACHE=/workspace/hf_datasets_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=disabled   # no wandb key — disable to avoid crash

# Step 1 — compute normalisation statistics (~40 min, only needed once per clone)
.venv/bin/python scripts/compute_norm_stats.py --config-name pi05_libero_lora

# Step 2 — fine-tune in background (logs to /workspace/training.log)
nohup bash -c '
export HF_DATASETS_CACHE=/workspace/hf_datasets_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/train.py pi05_libero_lora --exp-name=libero_lora_run1 --overwrite
' > /workspace/training.log 2>&1 &
echo "Training PID: $!"
```

Monitor progress (loss is logged every 100 steps):
```bash
tail -f /workspace/training.log | grep -E "Step [0-9]+:"
```

## Resuming an interrupted training run

Drop `--overwrite` to resume from the last checkpoint:

```bash
nohup bash -c '
export HF_DATASETS_CACHE=/workspace/hf_datasets_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/train.py pi05_libero_lora --exp-name=libero_lora_run1 --resume
' > /workspace/training.log 2>&1 &
```

## Output locations

| Item | Path |
|------|------|
| Norm stats | `~/libero-finetune/openpi/assets/pi05_libero_lora/physical-intelligence/libero/norm_stats.json` (backed up to `/workspace/openpi_assets/`) |
| Checkpoints | `~/libero-finetune/openpi/checkpoints/pi05_libero_lora/libero_lora_run1/` |
| Training log | `/workspace/training.log` |

Checkpoints are saved every 5000 steps (max_to_keep=1, plus permanent saves at multiples of 5000).

## Performance on RTX 4090 (24 GB)

- batch_size=24 fits with optimized config; batch_size=32 OOMs
- ~1.5 s/step after JIT warmup at batch_size=24
- 5,000 steps ≈ 2.5 hours total

## Convergence

With optimized hyperparameters (alpha 4x, batch_size=24, cosine decay matched to training length), loss plateaus around **0.022** by step ~4k:

```
Step 4300: grad_norm=0.0892, loss=0.0220
Step 4500: grad_norm=0.0955, loss=0.0225
Step 4700: grad_norm=0.0918, loss=0.0224
Step 4900: grad_norm=0.0911, loss=0.0223
```

**5k steps is sufficient.** See OPTIMIZATIONS.md for the full experiment log.

## Eval results

Evaluated on all 4 LIBERO benchmark suites (10 tasks x 50 trials each):

| Suite | Base model (no fine-tune) | Our LoRA @ 5k steps | pi0.5 @ 30k (full fine-tune, README) |
|-------|--------------------------|---------------------|--------------------------------------|
| libero_spatial | 0.0% | 95.8% | 98.8% |
| libero_object | 0.0% | **99.0%** | 98.2% |
| libero_goal | 0.0% | 91.6% | 98.0% |
| libero_10 | 0.0% | 86.2% | 92.4% |
| **Average** | **0.0%** | **93.15%** | **96.85%** |

The base model scores 0% — it has never seen LIBERO data and its action outputs are meaningless for this task space. Our LoRA fine-tune achieves **93.15% average**, within 3.7 points of the full fine-tune benchmark while using 6x fewer steps on a single RTX 4090. We beat the benchmark on libero_object.

## Key design notes

- `paligemma_variant="gemma_2b_lora"`: LoRA rank=16, alpha=64 (4x default) on attention + FFN of the 2B VLM
- `action_expert_variant="gemma_300m_lora"`: LoRA rank=32, alpha=128 (4x default) on the 300M action expert
- `freeze_filter`: trains only LoRA weights — pattern `All(PathRegex('.*llm.*'), Not(PathRegex('.*lora.*')))`
- `extra_delta_transform=False`: matches pi0.5 convention (pi0 uses `True`)
- **Use `train.py` (JAX), not `train_pytorch.py`** — LoRA is not supported in the PyTorch path
- The pi05_base checkpoint is downloaded from GCS automatically at training start (~20 GB, cached to `~/.cache/openpi/`)

## Running evaluation

Eval requires a separate Python 3.8 venv (libero/robosuite depend on older packages). Setup once:

```bash
cd ~/libero-finetune/openpi
git submodule update --init --recursive
sudo apt-get install -y python3.8 python3.8-venv python3.8-dev cmake libegl1-mesa-dev

python3.8 -m venv examples/libero/.venv
source examples/libero/.venv/bin/activate
pip install --upgrade pip
pip install -r examples/libero/requirements.txt -r third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e packages/openpi-client -e third_party/libero
pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install egl-probe robosuite==1.4.1 bddl pyyaml easydict matplotlib cloudpickle \
  'gym==0.23.1' gymnasium imageio imageio-ffmpeg tyro
deactivate
```

Also create `~/.libero/config.yaml` (to skip interactive prompt):
```yaml
benchmark_root: /home/ubuntu/libero-finetune/openpi/third_party/libero/libero/libero
bddl_files: /home/ubuntu/libero-finetune/openpi/third_party/libero/libero/libero/bddl_files
init_states: /home/ubuntu/libero-finetune/openpi/third_party/libero/libero/libero/init_files
datasets: /home/ubuntu/libero-finetune/openpi/third_party/libero/libero/datasets
assets: /home/ubuntu/libero-finetune/openpi/third_party/libero/libero/libero/assets
```

### Running eval (two processes)

**Terminal 1 — Policy server** (uses the training venv):
```bash
cd ~/libero-finetune/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
.venv/bin/python scripts/serve_policy.py --env LIBERO \
  policy:checkpoint --policy.config pi05_libero_lora \
  --policy.dir /workspace/checkpoints/pi05_libero_lora/optimized_run/4999
```

**Terminal 2 — Eval script** (uses the libero venv):
```bash
cd ~/libero-finetune/openpi
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH
MUJOCO_GL=egl python examples/libero/main.py --args.task-suite-name libero_spatial
```

Run for each suite: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`.
Each suite runs 10 tasks x 50 trials. Takes 40–90 minutes per suite depending on episode length.

## LIBERO dataset

- HuggingFace repo: `physical-intelligence/libero`
- Format: LeRobot v2.0 (parquet, images embedded — no separate video files)
- 1693 episodes, ~35 GB parquet + ~12 GB xet cache
- All episodes are in the `train` split
- Arrow cache (~10–15 GB) is built automatically on first run and cached in `HF_DATASETS_CACHE`
- LeRobot will warn about v2.0 vs v2.1 format — this is harmless, training works fine
