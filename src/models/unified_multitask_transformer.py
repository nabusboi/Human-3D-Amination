"""
Unified Multi-Task Transformer with Switchable Per-Task LoRA Adapters.

Architecture:
  - Shared CNN image encoder  (coco, faces tasks)
  - Shared MoCap encoder      (mocap task)
  - Shared Landmark encoder   (hdtf task)
  - Shared Transformer backbone with FROZEN base weights
  - Per-task LoRA A/B matrices stored inside each FFN linear layer
  - Per-task output heads

Only LoRA matrices + the relevant head are trained per task.
Everything else stays frozen → <5% trainable params per task.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS = ["coco", "mocap", "hdtf", "faces"]

TASK_OUTPUT_DIMS = {
    "coco":  17 * 3,   # 17 COCO keypoints  (x, y, vis)
    "mocap": 31 * 3,   # 31 CMU joints      (x, y, z)
    "hdtf":  4,        # talking head params (mouth, lip, blink, roll)
    "faces": 25,       # camera pose matrix  (16) + intrinsics (9)
}

TASK_INPUT_TYPES = {
    "coco":  "image",     # [B, 3, H, W]
    "mocap": "mocap",     # [B, seq_len, 31, 3]
    "hdtf":  "landmark",  # [B, 11, 3]
    "faces": "image",     # [B, 3, H, W]
}


# ---------------------------------------------------------------------------
# Switchable LoRA Linear — one A/B pair per task, all stored in the layer
# ---------------------------------------------------------------------------

class SwitchableLoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with per-task LoRA adapters.
    Base weight is frozen. Only the selected task's A/B matrices are updated.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 tasks: list = TASKS,
                 r: int = 16,
                 alpha: int = 32,
                 dropout: float = 0.05,
                 bias: bool = True):
        super().__init__()
        self.r = r
        self.scaling = alpha / r

        # Frozen base linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_param = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.requires_grad_(False)
        if self.bias_param is not None:
            self.bias_param.requires_grad_(False)

        # Per-task LoRA: A initialized with kaiming, B with zeros (standard LoRA init)
        self.lora_A = nn.ParameterDict({
            t: nn.Parameter(torch.empty(r, in_features)) for t in tasks
        })
        self.lora_B = nn.ParameterDict({
            t: nn.Parameter(torch.zeros(out_features, r)) for t in tasks
        })
        for t in tasks:
            nn.init.kaiming_uniform_(self.lora_A[t], a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, task: Optional[str] = None) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias_param)
        if task is not None and task in self.lora_A:
            lora = F.linear(
                F.linear(self.dropout(x), self.lora_A[task]),
                self.lora_B[task]
            )
            return base + lora * self.scaling
        return base


# ---------------------------------------------------------------------------
# Switchable Transformer Encoder Layer
# ---------------------------------------------------------------------------

class SwitchableTransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer where FFN linears carry per-task LoRA adapters."""

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int,
                 tasks: list = TASKS,
                 r: int = 16,
                 alpha: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Switchable LoRA feedforward
        self.ff1 = SwitchableLoRALinear(d_model, dim_feedforward, tasks, r, alpha, dropout)
        self.ff2 = SwitchableLoRALinear(dim_feedforward, d_model, tasks, r, alpha, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, task: Optional[str] = None) -> torch.Tensor:
        # Multi-head self-attention (shared, no LoRA — MHA internals access .weight directly)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        # FFN with switchable LoRA
        ff_out = self.ff2(self.drop(self.act(self.ff1(x, task))), task)
        x = self.norm2(x + self.drop(ff_out))
        return x


# ---------------------------------------------------------------------------
# Unified Multi-Task Transformer
# ---------------------------------------------------------------------------

class UnifiedMultiTaskTransformer(nn.Module):
    """
    Single model that handles COCO pose, CMU MoCap, HDTF facial, and Synthetic Faces tasks.

    Shared:
      - Feature encoders (image CNN, mocap linear, landmark linear)
      - Positional embedding
      - Transformer backbone (base weights frozen after init)

    Per-task (trained with LoRA, <5% params):
      - LoRA A/B matrices in each FFN layer
      - Task-specific output head
    """

    def __init__(self,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 r: int = 16,
                 alpha: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # --- Input Encoders ---
        # Image encoder for coco + faces: [B, 3, H, W] → [B, D, h, w]
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim), nn.GELU()
        )

        # MoCap encoder for mocap: [B, seq, 93] → [B, seq, D]
        self.mocap_encoder = nn.Linear(31 * 3, embed_dim)

        # Landmark encoder for hdtf: [B, 11, 3] → [B, 11, D]
        self.landmark_encoder = nn.Linear(3, embed_dim)

        # --- Positional Embedding (shared, up to 512 tokens) ---
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # --- Switchable Transformer Encoder ---
        self.transformer_layers = nn.ModuleList([
            SwitchableTransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                tasks=TASKS, r=r, alpha=alpha,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # --- Task-Specific Output Heads ---
        self.task_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.GELU(),
                nn.Linear(128, out_dim)
            )
            for task, out_dim in TASK_OUTPUT_DIMS.items()
        })

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor, task: str) -> torch.Tensor:
        """Route input through the correct encoder → [B, L, D]."""
        if task in ("coco", "faces"):
            feats = self.image_encoder(x)          # [B, D, h, w]
            _, _, h, w = feats.shape
            return feats.flatten(2).transpose(1, 2)  # [B, h*w, D]
        elif task == "mocap":
            if x.dim() == 4:                        # [B, T, 31, 3]
                b, t, j, d = x.shape
                x = x.reshape(b, t, j * d)
            return self.mocap_encoder(x)             # [B, T, D]
        elif task == "hdtf":
            return self.landmark_encoder(x)          # [B, 11, D]
        else:
            raise ValueError(f"Unknown task '{task}'. Valid: {TASKS}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, task: str = "coco") -> torch.Tensor:
        b = x.shape[0]

        # Encode input to sequence
        seq = self._encode(x, task)                     # [B, L, D]
        L = seq.shape[1]

        # Add positional embedding — interpolate if sequence is longer than base size
        L = seq.shape[1]
        if L <= self.pos_embed.shape[1]:
            pos = self.pos_embed[:, :L, :]
        else:
            # Dynamically resize pos_embed to match actual sequence length
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2),   # [1, D, base_len]
                size=L, mode="linear", align_corners=False
            ).transpose(1, 2)                      # [1, L, D]
        seq = seq + pos

        # Transformer with task-specific LoRA adapters
        for layer in self.transformer_layers:
            seq = layer(seq, task=task)

        # Global average pool
        pooled = seq.mean(dim=1)                        # [B, D]
        raw = self.task_heads[task](pooled)

        # Reshape to task-specific output
        if task == "coco":
            return raw.view(b, 17, 3)
        elif task == "mocap":
            return raw.view(b, 31, 3)
        else:
            return raw                                  # [B, 4] or [B, 25]

    # ------------------------------------------------------------------
    # LoRA task switching utilities
    # ------------------------------------------------------------------

    def freeze_all_except_task(self, task: str):
        """
        Freeze everything except:
          - LoRA A/B matrices for `task` in every FFN layer
          - The task-specific output head
        """
        for name, p in self.named_parameters():
            is_lora = (f"lora_A.{task}" in name or f"lora_B.{task}" in name)
            is_head = name.startswith(f"task_heads.{task}.")
            p.requires_grad_(is_lora or is_head)

    def count_trainable(self) -> tuple:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def print_trainable(self, task: str = None):
        t, total = self.count_trainable()
        label = f"[Task: {task}] " if task else ""
        print(f"{label}trainable params: {t:,} || all params: {total:,} || "
              f"trainable%: {100 * t / total:.4f}")
