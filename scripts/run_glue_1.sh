#!/usr/bin/env bash
set -euo pipefail

# Simple experiment launcher for run_glue.py
# Varies batch size and gradient accumulation steps and writes outputs to separate dirs.

PYTHON=${PYTHON:-python3}
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$SCRIPT_ROOT/run_glue_custom.py"

# configure these arrays to run experiments
BATCH_SIZES=(1 8 16 32 128)
ACC_STEPS=(1)
# PROJ_GAP=(100 300 500 1000)

# other defaults (tweak as needed)
EPOCHS=${EPOCHS:-30}
GPU_NUM=${GPU_NUM:-1}
TASK_NAME=${TASK_NAME:-mrpc}

OUT_ROOT="$SCRIPT_ROOT/exp_outputs/${TASK_NAME}_base"
mkdir -p "$OUT_ROOT"

for BS in "${BATCH_SIZES[@]}"; do
  for ACC in "${ACC_STEPS[@]}"; do
    NAME="bs${BS}_acc${ACC}_epochs${EPOCHS}"
    OUT_DIR="$OUT_ROOT/$NAME"
    mkdir -p "$OUT_DIR"

    echo "Running experiment: batch_size=$BS grad_accum_steps=$ACC -> output=$OUT_DIR"

    CMD=("$PYTHON" "$SCRIPT"
        --model_name_or_path roberta-base \
        --task_name "$TASK_NAME" \
        --enable_galore \
        --lora_all_modules \
        --max_length 512 \
        --seed 1234 \
        --gradient_accumulation_steps "$ACC" \
        --lora_r 4 \
        --galore_scale 4 \
        --per_device_train_batch_size "$BS" \
        --update_proj_gap 500 \
        --learning_rate 3e-5 \
        --num_train_epochs "$EPOCHS" \
        --output_dir "$OUT_DIR"
        # --enable_single_projection
    )

    echo "Command: ${CMD[*]}"

    # run and capture logs (set CUDA_VISIBLE_DEVICES in the subshell env)
    (
      cd "$SCRIPT_ROOT"
      CUDA_VISIBLE_DEVICES="$GPU_NUM" "${CMD[@]}" 2>&1 | tee "$OUT_DIR/run.log"
    )

  done
done

echo "All experiments completed. Outputs under: $OUT_ROOT"
