"""
Synthetic Faces Dataset (EG3D Synthetic Human Faces for 3D Reconstruction).
Pairs PNG face images with camera pose .npy files.
Target: predict camera pose matrix [25] from face image.
"""

import os
import glob
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from typing import List, Tuple, Optional
import config


class SyntheticFacesDataset(Dataset):
    """
    Dataset for the Kaggle 'Synthetic Human Faces for 3D Reconstruction' dataset.

    Directory structure:
      data/synthetic_faces/
        eg3d_synthetic_faces/eg3d_synthetic_faces/  ← PNG images (seed0000.png ...)
        cam_poses-*/cam_poses/                       ← camera pose NPY files (seed0000.npy ...)

    Returns:
        image_tensor: [3, 256, 256]
        cam_pose:     [25]  (camera extrinsics 4×4 + intrinsics 3×3 + padding, flattened)
    """

    def __init__(self,
                 data_dir: str = config.DATASET_SUBDIRS["synthetic_faces"],
                 image_size: Tuple[int, int] = (256, 256)):
        self.image_size = image_size
        self.samples: List[Tuple[str, str]] = []   # (img_path, npy_path) pairs
        self.synthetic = True

        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if os.path.exists(data_dir):
            self._discover(data_dir)
            if not self.samples:
                # Try parent directory if data_dir didn't yield matches
                parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(data_dir)))
                if os.path.exists(parent_dir):
                    self._discover(parent_dir)

        if not self.samples:
            print(f"[Synthetic Faces] No image/pose pairs found. Using synthetic fallback.")
        else:
            self.synthetic = False
            print(f"[Synthetic Faces] Loaded {len(self.samples)} image/cam-pose pairs.")

    def _discover(self, root: str):
        """
        Find all seed????.png images and match them to corresponding seed????.npy cam poses or latents.
        """
        # Find image files
        img_files = glob.glob(os.path.join(root, "**", "seed*.png"), recursive=True)

        # Find all npy files keyed by seed number (search in root or parent)
        search_base = root if "synthetic_faces" in os.path.basename(root) else os.path.dirname(root)
        npy_files = glob.glob(
            os.path.join(search_base, "**", "seed*.npy"), recursive=True
        )
        npy_map = {}
        for npy in npy_files:
            m = re.search(r"seed(\d+)\.npy", os.path.basename(npy))
            if m:
                npy_map[int(m.group(1))] = npy

        for img in img_files:
            m = re.search(r"seed(\d+)\.png", os.path.basename(img))
            if m:
                seed = int(m.group(1))
                if seed in npy_map:
                    self.samples.append((img, npy_map[seed]))

        self.samples.sort(key=lambda x: x[0])

    def _load_cam_pose(self, npy_path: str) -> np.ndarray:
        """Load and flatten camera pose. Pads/truncates to 25 values."""
        try:
            data = np.load(npy_path).astype(np.float32).flatten()
            out = np.zeros(25, dtype=np.float32)
            n = min(len(data), 25)
            out[:n] = data[:n]
            return out
        except Exception:
            return np.zeros(25, dtype=np.float32)

    def __len__(self) -> int:
        return max(100, len(self.samples))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.synthetic:
            img_tensor = torch.randn(3, *self.image_size)
            cam_pose = torch.randn(25)
            return img_tensor, cam_pose

        img_path, npy_path = self.samples[idx % len(self.samples)]
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img)
        except Exception:
            img_tensor = torch.randn(3, *self.image_size)

        cam_pose = torch.tensor(self._load_cam_pose(npy_path), dtype=torch.float32)
        return img_tensor, cam_pose
