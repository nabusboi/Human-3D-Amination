import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from typing import Optional, Tuple, List, Dict
import config


class COCOPersonDataset(Dataset):
    """
    PyTorch Dataset Reader for Microsoft COCO 2017 Keypoint Annotations.
    Works directly with unzipped Kaggle COCO 2017 datasets (28 GB full or 1 GB val).
    """

    def __init__(self,
                 data_dir: str = config.KAGGLE_DATASET_DIR,
                 split: str = "val2017",
                 image_size: Tuple[int, int] = (256, 256)):
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size

        self.img_dir = os.path.join(data_dir, split)
        self.ann_file = os.path.join(data_dir, "annotations", f"person_keypoints_{split}.json")

        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.samples: List[Dict] = []
        self.has_real_data = False

        if os.path.exists(self.ann_file) and os.path.exists(self.img_dir):
            self._load_coco_annotations()
        else:
            print(f"[COCO Dataset Note] COCO annotations file not found at: {self.ann_file}.")
            print("Using synthetic dataset fallback for pipeline testing until Kaggle dataset path is provided.")
            self._create_synthetic_samples()

    def _load_coco_annotations(self):
        """Parses COCO JSON annotations for person instances with valid keypoints."""
        try:
            with open(self.ann_file, "r") as f:
                data = json.load(f)

            images_dict = {img["id"]: img for img in data.get("images", [])}
            
            for ann in data.get("annotations", []):
                # Filter for person category (id 1 in COCO) with >0 keypoints
                if ann.get("category_id") == 1 and ann.get("num_keypoints", 0) > 0:
                    img_id = ann.get("image_id")
                    img_info = images_dict.get(img_id)
                    if img_info:
                        img_path = os.path.join(self.img_dir, img_info["file_name"])
                        keypoints = ann.get("keypoints", [])  # Flat list of 51 items [x1, y1, v1, ...]
                        bbox = ann.get("bbox", [])            # [x, y, width, height]
                        
                        if len(keypoints) == 51 and os.path.exists(img_path):
                            self.samples.append({
                                "img_path": img_path,
                                "keypoints": keypoints,
                                "bbox": bbox,
                                "width": img_info["width"],
                                "height": img_info["height"]
                            })

            self.has_real_data = len(self.samples) > 0
            print(f"[COCO Dataset] Loaded {len(self.samples)} person keypoint samples from {self.split}.")
        except Exception as e:
            print(f"[COCO Dataset Error] Failed to parse COCO json: {e}")
            self._create_synthetic_samples()

    def _create_synthetic_samples(self, num_samples: int = 100):
        """Generates synthetic 17-keypoint samples for offline testing."""
        self.samples = []
        for i in range(num_samples):
            self.samples.append({
                "synthetic": True,
                "width": 640,
                "height": 480
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        if sample.get("synthetic", False) or not self.has_real_data:
            # Generate synthetic image tensor & keypoints tensor [17, 3]
            img_tensor = torch.randn(3, self.image_size[0], self.image_size[1])
            keypoints_tensor = torch.rand(17, 3)
            keypoints_tensor[:, :2] *= float(self.image_size[0])
            keypoints_tensor[:, 2] = 2.0  # Visible
            return img_tensor, keypoints_tensor

        # Load real image
        img_path = sample["img_path"]
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = sample["width"], sample["height"]

        # Parse 17 keypoints [17, 3]
        raw_kp = np.array(sample["keypoints"], dtype=np.float32).reshape(17, 3)
        
        # Scale keypoint coordinates to resized target image dimensions
        scale_x = self.image_size[1] / orig_w
        scale_y = self.image_size[0] / orig_h

        scaled_kp = raw_kp.copy()
        scaled_kp[:, 0] *= scale_x
        scaled_kp[:, 1] *= scale_y

        img_tensor = self.transform(img)
        keypoints_tensor = torch.tensor(scaled_kp, dtype=torch.float32)

        return img_tensor, keypoints_tensor
