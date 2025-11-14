#!/usr/bin/env bash
set -euo pipefail

# Simple experiment launcher for run_CIFAR100.py
# Varies batch size and gradient accumulation steps and writes outputs to separate dirs.

PYTHON=${PYTHON:-python3}
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$SCRIPT_ROOT/run_CIFAR100.py"

# configure these arrays to run experiments
BATCH_SIZES=(8 16 32 64 128)
ACC_STEPS=(1)

# other defaults (tweak as needed)
EPOCHS=${EPOCHS:-30}
LR=${LR:-5e-5}
MODEL_NAME=${MODEL_NAME:-google/vit-base-patch16-224-in21k}
USE_AMP=${USE_AMP:-false}
GPU_NUM=${GPU_NUM:-1}

OUT_ROOT="$SCRIPT_ROOT/exp_outputs"
mkdir -p "$OUT_ROOT"

for BS in "${BATCH_SIZES[@]}"; do
  for ACC in "${ACC_STEPS[@]}"; do
    NAME="bs${BS}_acc${ACC}_epochs${EPOCHS}"
    OUT_DIR="$OUT_ROOT/$NAME"
    mkdir -p "$OUT_DIR"

    echo "Running experiment: batch_size=$BS grad_accum_steps=$ACC -> output=$OUT_DIR"

    CMD=("$PYTHON" "$SCRIPT"
      --model_name "$MODEL_NAME"
      --batch_size "$BS"
      --grad_accum_steps "$ACC"
      --epochs "$EPOCHS"
      --lr "$LR"
      --output_dir "$OUT_DIR"
      --warmup_steps 0
      --save_every 1
      --rank 4
      --update_proj_gap 500
      --galore_scale 4
    )

    if [ "$USE_AMP" = true ]; then
      CMD+=(--use_amp)
    fi

    echo "Command: ${CMD[*]}"

    # run and capture logs (set CUDA_VISIBLE_DEVICES in the subshell env)
    (
      cd "$SCRIPT_ROOT"
      CUDA_VISIBLE_DEVICES="$GPU_NUM" "${CMD[@]}" 2>&1 | tee "$OUT_DIR/run.log"
    )

  done
done

echo "All experiments completed. Outputs under: $OUT_ROOT"
