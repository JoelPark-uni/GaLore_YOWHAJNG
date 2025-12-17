# GaLore_YOWHAJNG Experiment Guide

This repository contains GaLore experiments on GLUE (MRPC) and CIFAR-100. Five experiment types are set up with ready-to-run scripts and analysis utilities.

## Environment
- Python 3.10+ recommended. Install dependencies from the repo root:
  ```bash
  pip install -r requirements.txt
  # or use exp_requirements.txt when reproducing paper configs
  ```
- Hugging Face datasets will download to the default cache; ensure network access the first time MRPC or CIFAR-100 is used.

## 1) MRPC Batch Size & Gradient Accumulation Sweeps
- Purpose: compare MRPC performance across batch sizes and grad-acc settings using GaLore-enhanced training.
- Scripts: run the prepared command lists in [scripts/run_glue_batch_size.sh](scripts/run_glue_batch_size.sh) and [scripts/run_glue_acc_steps.sh](scripts/run_glue_acc_steps.sh).
  ```bash
  # vary batch sizes (examples for bs=128,16,1)
  bash scripts/run_glue_batch_size.sh

  # vary accumulation steps (examples for acc=8,2,1)
  bash scripts/run_glue_acc_steps.sh
  ```
- Each command calls [run_glue.py](run_glue.py) with `task_name=mrpc`, GaLore enabled, and writes outputs under `results/ft/roberta_base_mrpc_T10/mrpc*`. Edit the files to change `per_device_train_batch_size`, `gradient_accumulation_steps`, or output paths.

## 2) Stable Rank Experiments
- Compute stable rank of gradients for MRPC-trained checkpoints and plot trends.
- Collect metrics: adjust `MODEL_SPECS` and `BATCH_SIZES` in [gradient_rank/gradient_rank.py](gradient_rank/gradient_rank.py) and run:
  ```bash
  python gradient_rank/gradient_rank.py
  ```
  JSON files are saved to `gradient_rank/stable_rank/` (one per model id and batch size).
- Visualize: run [gradient_rank/gradient_graph.py](gradient_rank/gradient_graph.py) to aggregate means (overall and per layer) and emit plots into `gradient_rank/plots3/`.

## 3) Single-Sample Projection (GaLore)
- Purpose: project gradients using only the first sample of each batch.
- Launcher: [scripts/run_glue_single_sample.sh](scripts/run_glue_single_sample.sh) wraps [run_glue_custom.py](run_glue_custom.py).
  ```bash
  bash scripts/run_glue_single_sample.sh
  ```
- The script defaults to MRPC with Roberta-base; uncomment `--enable_single_projection` inside the command to activate single-sample GaLore projection. Outputs land under `exp_outputs/mrpc_FT/bs<...>/acc<...>/` (one directory per setting).

## 4) Training Loss Stability Notebook
- Notebook: [notebooks/mrpc_batchsize_analysis.ipynb](notebooks/mrpc_batchsize_analysis.ipynb).
- The third section (“Training Loss Stability”) compares loss traces across batch sizes. Open the notebook in VS Code/Jupyter, run the initial setup cells to load logged metrics, then execute the cells in that section to reproduce the plots.

## 5) CIFAR-100 Experiments
- Script: [scripts/run_CIFAR100.sh](scripts/run_CIFAR100.sh) sweeps batch sizes (4–128) with gradient accumulation on ViT.
  ```bash
  bash scripts/run_CIFAR100.sh
  ```
- Internals: calls [run_CIFAR100.py](run_CIFAR100.py) (ViT fine-tuning) with GaLore optimizer settings (`rank`, `update_proj_gap`, `galore_scale`). Set `USE_AMP=true` to enable mixed precision. Results and logs are written to `exp_outputs/bs<...>_acc<...>_epochs<...>/`.

## Tips
- Override `GPU_NUM`, `BATCH_SIZES`, `ACC_STEPS`, or learning rates by editing the launcher scripts. Logs are tee’d into each experiment’s output directory for reproducibility.
- For custom MRPC runs without the helper scripts:
  ```bash
  python run_glue.py --model_name_or_path roberta-base --task_name mrpc \
    --enable_galore --lora_all_modules --lora_r 4 --galore_scale 4 \
    --update_proj_gap 10 --per_device_train_batch_size 32 --gradient_accumulation_steps 2 \
    --num_train_epochs 30 --max_length 512 --output_dir results/custom_run
  ```
