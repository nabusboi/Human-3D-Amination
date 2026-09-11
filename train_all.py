"""
train_all.py — Unified Multi-Task QLoRA Training Pipeline

Trains a single UnifiedMultiTaskTransformer on all four tasks sequentially:
  1. coco   → COCO 2017 human keypoint detection
  2. mocap  → CMU MoCap 3D motion forecasting
  3. hdtf   → HDTF facial talking head parameters
  4. faces  → Synthetic Faces camera pose regression

Each task trains ONLY its LoRA adapters + output head (<5% params).
All other weights stay frozen. Final checkpoint saves the entire model.

Usage:
    python train_all.py
    python train_all.py --epochs 15 --batch_size 32
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from src.models.unified_multitask_transformer import UnifiedMultiTaskTransformer
from src.dataset.coco_dataset import COCOPersonDataset
from src.dataset.cmu_mocap_dataset import CMUMoCapDataset
from src.dataset.hdtf_facial_dataset import HDTFFacialDataset
from src.dataset.synthetic_faces_dataset import SyntheticFacesDataset
from src.dataset.dataset_downloader import DatasetDownloader


# ---------------------------------------------------------------------------
# Wing Loss (high-precision facial landmark regression)
# ---------------------------------------------------------------------------

class WingLoss(nn.Module):
    def __init__(self, omega: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.c = omega - omega * torch.log(torch.tensor(1.0 + omega / epsilon))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = (pred - target).abs()
        loss = torch.where(
            delta < self.omega,
            self.omega * torch.log(1.0 + delta / self.epsilon),
            delta - self.c.to(pred.device)
        )
        return loss.mean()


# ---------------------------------------------------------------------------
# Per-task loss functions
# ---------------------------------------------------------------------------

def get_loss_fn(task: str) -> nn.Module:
    if task == "coco":
        return WingLoss()           # precision facial + Smooth L1 body
    elif task == "mocap":
        return nn.MSELoss()         # motion prediction
    elif task == "hdtf":
        return nn.SmoothL1Loss()    # talking head regression
    elif task == "faces":
        return nn.MSELoss()         # camera pose regression
    else:
        return nn.SmoothL1Loss()


# ---------------------------------------------------------------------------
# Per-task dataset factory
# ---------------------------------------------------------------------------

def get_dataset(task: str) -> torch.utils.data.Dataset:
    if task == "coco":
        return COCOPersonDataset(data_dir=config.KAGGLE_DATASET_DIR, split="val2017")
    elif task == "mocap":
        return CMUMoCapDataset(data_dir=config.DATASET_SUBDIRS["cmu_mocap"])
    elif task == "hdtf":
        return HDTFFacialDataset(data_dir=config.DATASET_SUBDIRS["hdtf"])
    elif task == "faces":
        return SyntheticFacesDataset(data_dir=config.DATASET_SUBDIRS["synthetic_faces"])
    else:
        raise ValueError(f"Unknown task: {task}")


# ---------------------------------------------------------------------------
# Training loop for a single task
# ---------------------------------------------------------------------------

def train_task(model: UnifiedMultiTaskTransformer,
               task: str,
               device: torch.device,
               epochs: int,
               batch_size: int,
               lr: float):
    print(f"\n{'=' * 60}")
    print(f"  TASK: {task.upper()}")
    print(f"{'=' * 60}")

    # Freeze all except task-specific LoRA + head
    model.freeze_all_except_task(task)
    model.print_trainable(task)

    # Dataset + DataLoader
    dataset = get_dataset(task)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Optimizer (only unfrozen params)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = get_loss_fn(task).to(device)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0

        for batch in loader:
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                preds = model(x, task=task)
                preds = torch.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=-1.0)

                # mocap target is [B, 31, 3]; coco target is [B, 17, 3]
                # hdtf target is [B, 4]; faces target is [B, 25]
                if task == "coco":
                    # Apply Wing loss on face KPs (0-4) + Smooth L1 on body (5-16)
                    face_loss = WingLoss()(preds[:, :5, :], y[:, :5, :])
                    body_loss = nn.SmoothL1Loss()(preds[:, 5:, :], y[:, 5:, :])
                    loss = face_loss * 1.5 + body_loss
                else:
                    loss = loss_fn(preds, y)

                loss = torch.nan_to_num(loss, nan=0.001, posinf=1.0, neginf=0.001)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())

        scheduler.step()
        avg_loss = running_loss / max(1, len(loader))
        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch [{epoch:02d}/{epochs}] | Loss: {avg_loss:.4f} | LR: {curr_lr:.6f}")

    return avg_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_all(epochs: int = config.EPOCHS,
              batch_size: int = config.BATCH_SIZE,
              lr: float = config.LEARNING_RATE,
              tasks: list = None):
    if tasks is None:
        tasks = ["coco", "mocap", "hdtf", "faces"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = DatasetDownloader.get_trained_model_dir("qlora_pose")
    save_path = os.path.join(save_dir, "unified_multitask_qlora.pt")

    print("=" * 60)
    print("   UNIFIED MULTI-TASK QLORA TRAINING PIPELINE")
    print("=" * 60)
    print(f"[Config] Device       : {device}")
    print(f"[Config] Tasks        : {tasks}")
    print(f"[Config] Epochs/task  : {epochs}")
    print(f"[Config] Batch size   : {batch_size}")
    print(f"[Config] Learning rate: {lr}")
    print(f"[Config] Save path    : {save_path}")
    print("=" * 60)

    # Ensure YOLO weights exist (used by the main app)
    DatasetDownloader.ensure_yolo_weights()

    # Build unified model
    model = UnifiedMultiTaskTransformer(
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        r=config.LORA_R,
        alpha=config.LORA_ALPHA,
        dropout=0.1
    ).to(device)

    task_losses = {}
    for task in tasks:
        final_loss = train_task(model, task, device, epochs, batch_size, lr)
        task_losses[task] = final_loss

    # Save full checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "tasks_trained": tasks,
        "task_losses": task_losses,
        "epochs": epochs,
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 4,
        "lora_r": config.LORA_R,
        "lora_alpha": config.LORA_ALPHA,
    }, save_path)

    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE -- All tasks done!")
    print(f"{'=' * 60}")
    for task, loss in task_losses.items():
        print(f"  [{task:6s}] Final loss: {loss:.4f}")
    print(f"\n  Checkpoint saved -> {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Multi-Task QLoRA Training.")
    parser.add_argument("--epochs",     type=int,   default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=config.LEARNING_RATE)
    parser.add_argument("--tasks",      type=str,   default="coco,mocap,hdtf,faces",
                        help="Comma-separated list of tasks to train")
    args = parser.parse_args()

    train_all(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        tasks=args.tasks.split(",")
    )
