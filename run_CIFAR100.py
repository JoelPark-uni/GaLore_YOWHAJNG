"""CIFAR-100 training script using Hugging Face Transformers ViT.

Major capabilities:
- seed_everything(seed): deterministic seeding for reproducibility
- parse_args(): CLI for hyperparameters and training options
- train() and evaluate() functions for training loop and validation

Usage (example):
    python run_CIFAR100.py --model_name google/vit-base-patch16-224-in21k --batch_size 64 --epochs 5

The script uses torchvision to load CIFAR-100 and a pretrained ViT model
from the `transformers` library for image classification.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
import logging
from datetime import datetime
from tqdm.auto import tqdm as tqdmrange

from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    get_scheduler,
)

from galore_torch import GaLoreAdamW


def seed_everything(seed: int) -> None:
    """Seed Python, numpy and torch (both CPU and CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducible cudnn (may slow down training)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ViT on CIFAR-100 using transformers")
    p.add_argument("--model_name", type=str, default="google/vit-base-patch16-224-in21k", help="pretrained ViT model name or path")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="./output_vit_cifar100")
    p.add_argument("--device", type=str, default='cuda', help="cuda or cpu (auto-detected if not set)")
    p.add_argument("--use_amp", action="store_true", help="Use mixed precision training (fp16)")
    p.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    
    p.add_argument("--rank", type=int, default=4, help="Rank for GaLore optimizer")
    p.add_argument("--update_proj_gap", type=int, default=500, help="Update projection gap for GaLore optimizer")
    p.add_argument("--galore_scale", type=float, default=4, help="Scale for GaLore optimizer")

    # p.add_argument("--gpu_num", type=int, default=0, help="GPU number to use")

    args = p.parse_args()
    return args


def build_dataloaders(image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    """Create CIFAR-100 train and validation dataloaders with transforms compatible with ViT."""
    # CIFAR images are 32x32; ViT pretrained on 224x224 -> resize
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    data_root = os.path.expanduser("~/.cache/cifar")
    train_ds = datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_transform)
    val_ds = datasets.CIFAR100(root=data_root, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler,
          device: torch.device,
          epochs: int,
          output_dir: str,
          use_amp: bool = False,
          save_every: int = 1,
          grad_accum_steps: int = 1) -> None:
    """Full training loop with optional mixed precision and periodic evaluation/save."""
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None

    best_acc = 0.0
    global_step = 0

    # metrics container
    metrics = {
        "train_loss_per_iteration": [],
        "epoch_metrics": []  # list of dicts with epoch-level metrics
    }
    metrics_path = os.path.join(output_dir, "metrics.json")

    # logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("train")

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        # zero grads before starting accumulation
        optimizer.zero_grad()

        for batch in tqdmrange(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(pixel_values=imgs, labels=labels)
                    loss = outputs.loss
                # scale loss for accumulation
                scaled_loss = loss / float(grad_accum_steps)
                scaler.scale(scaled_loss).backward()
            else:
                outputs = model(pixel_values=imgs, labels=labels)
                loss = outputs.loss
                (loss / float(grad_accum_steps)).backward()

            # perform optimizer step only every grad_accum_steps
            if (global_step + 1) % grad_accum_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                # zero grads after step
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

            epoch_loss += loss.item()
            # record loss per iteration
            metrics["train_loss_per_iteration"].append({
                "epoch": epoch,
                "step": global_step,
                "loss": loss.item(),
            })
            logits = outputs.logits.detach()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            global_step += 1

        epoch_time = time.time() - start_time
        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)

        logger.info(f"Epoch {epoch}/{epochs} - time: {epoch_time:.1f}s - train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - val_f1: {val_f1:.4f}")

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(os.path.join(output_dir, "best_model"))
            logger.info(f"Saved best model with val_acc={best_acc:.4f}")

        # periodic save
        if epoch % save_every == 0:
            ckpt_dir = os.path.join(output_dir, f"checkpoint-epoch{epoch}")
            model.save_pretrained(ckpt_dir)

        # record epoch metrics and flush to disk
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        metrics["epoch_metrics"].append(epoch_record)

        metrics_path = os.path.join(output_dir, "metrics.json")
        try:
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception:
            logger.exception("Failed to write metrics to json")

    logger.info(f"Training complete. Best val_acc={best_acc:.4f}. Metrics saved to {metrics_path}")


def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    """Evaluate model on validation set and return (loss, accuracy, f1_macro)."""
    was_training = model.training
    model.eval()
    model.to(device)

    total = 0
    correct = 0
    losses = 0.0
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(pixel_values=imgs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            losses += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

    avg_loss = losses / len(val_loader)
    acc = correct / total

    # compute macro F1
    val_f1 = 0.0
    
    from sklearn.metrics import f1_score

    if len(labels_all) > 0:
        val_f1 = float(f1_score(labels_all, preds_all, average="macro"))
    
    if was_training:
        model.train()

    return avg_loss, acc, val_f1


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)

    # device
    # device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # processor to get image size & normalization if available
    try:
        image_processor = AutoImageProcessor.from_pretrained(args.model_name, use_fast=True)
        # many processors provide size as 'crop_size' or feature_extractor.size
        image_size = getattr(image_processor, "size", None)
        if image_size is None:
            image_size = getattr(image_processor, "crop_size", None)
        if isinstance(image_size, dict):
            # some processors store size as {"height": 224, "width": 224}
            image_size = image_size.get("height", 224)
        if image_size is None:
            image_size = 224
        if isinstance(image_size, (tuple, list)):
            image_size = int(image_size[0])
    except Exception:
        print("Warning: could not load AutoImageProcessor, falling back to 224")
        image_size = 224

    train_loader, val_loader = build_dataloaders(image_size=image_size,
                                                 batch_size=args.batch_size,
                                                 num_workers=args.num_workers)

    # load model and adapt classification head
    model = ViTForImageClassification.from_pretrained(args.model_name, num_labels=100)

    # optimizer and scheduler
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # label layers for galore optimizer
    # target_modules_list = ["attn", "mlp"]
    # target_modules_list = ["q_proj", "v_proj"]
    target_modules_list = ["attention", "dense"]
    galore_params = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # print('checking module: ', module_name)
        if not any(target_key in module_name for target_key in target_modules_list):
            continue

        print('enable GaLore for weights in module: ', module_name)
        galore_params.append(module.weight)

    id_galore_params = [id(p) for p in galore_params]
    # make parameters without "rank" to another group
    regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
    # then call galore_adamw
    param_groups = [{'params': regular_params}, 
                    {'params': galore_params, 'rank': args.rank, 'update_proj_gap': args.update_proj_gap, 'scale': args.galore_scale, 'proj_type': "std"}]
    optimizer = GaLoreAdamW(param_groups, lr=args.lr)
    

    total_steps = args.epochs * len(train_loader)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    train(model=model,
          train_loader=train_loader,
          val_loader=val_loader,
          optimizer=optimizer,
          scheduler=scheduler,
          device=device,
          epochs=args.epochs,
          output_dir=args.output_dir,
          use_amp=args.use_amp,
          save_every=args.save_every)


if __name__ == "__main__":
    main()
