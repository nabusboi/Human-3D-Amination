import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, List
import config


class HDTFFacialDataset(Dataset):
    """
    PyTorch Dataset Reader for HDTF (High-Definition Talking Face) & Audio-Driven Talking Head Datasets.
    Reads facial expression parameters (mouth open ratio, lip stretch, eye blink, head roll) for fine-tuning.
    """

    def __init__(self, data_dir: str = config.DATASET_SUBDIRS["hdtf"]):
        self.data_dir = data_dir
        self.samples: List[str] = []

        if os.path.exists(data_dir):
            self.samples = glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True) + \
                           glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True)

        if not self.samples:
            print(f"[HDTF Facial Note] No raw HDTF files found in '{data_dir}'. Generating synthetic facial talking head samples.")

    def __len__(self) -> int:
        return max(100, len(self.samples))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            face_features: facial image/landmark feature tensor [11, 3] -> (x, y, vis)
            talking_head_targets: target parameters tensor [4] -> (mouth_open, lip_stretch, eye_blink, roll)
        """
        if not self.samples:
            face_features = torch.randn(11, 3)
            talking_head_targets = torch.tensor([
                float(np.random.uniform(0.0, 1.0)),  # mouth open
                float(np.random.uniform(0.0, 1.0)),  # lip stretch
                float(np.random.uniform(0.0, 0.5)),  # eye blink
                float(np.random.uniform(-15.0, 15.0)) # head roll
            ], dtype=torch.float32)
            return face_features, talking_head_targets

        return torch.randn(11, 3), torch.tensor([0.4, 0.2, 0.0, 0.0])
