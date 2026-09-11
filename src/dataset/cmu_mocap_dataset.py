"""
CMU MoCap Dataset with full ASF/AMC parser.
Reads raw .asf skeleton + .amc motion files from the CMU Motion Capture database.
Returns (input_sequence [seq_len, 31, 3], next_frame_target [31, 3]) per sample.
Falls back to synthetic sinusoidal motion if no files found.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import config


# ---------------------------------------------------------------------------
# ASF / AMC Parsers
# ---------------------------------------------------------------------------

def parse_asf(filepath: str) -> Tuple[List[str], Dict]:
    """
    Parse a VICON .asf skeleton file.
    Returns:
        bone_names: ordered list of bone names (root first)
        bone_dofs:  dict mapping bone_name → list of active DOF channels
    """
    bone_names = ["root"]
    bone_dofs: Dict[str, List[str]] = {"root": ["tx", "ty", "tz", "rx", "ry", "rz"]}

    in_bonedata = False
    current: Dict = {}

    try:
        with open(filepath, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith(":bonedata"):
                    in_bonedata = True
                    continue
                if line.startswith(":") and in_bonedata:
                    in_bonedata = False
                if in_bonedata:
                    if line == "begin":
                        current = {}
                    elif line == "end" and current:
                        name = current.get("name")
                        if name:
                            bone_names.append(name)
                            bone_dofs[name] = current.get("dof", ["rx", "ry", "rz"])
                        current = {}
                    elif current is not None:
                        parts = line.split()
                        if not parts:
                            continue
                        key = parts[0]
                        if key == "name":
                            current["name"] = parts[1]
                        elif key == "dof":
                            current["dof"] = parts[1:]
    except Exception as e:
        print(f"[CMU ASF] Warning parsing {filepath}: {e}")

    return bone_names, bone_dofs


def parse_amc(filepath: str,
              bone_names: List[str],
              bone_dofs: Dict[str, List[str]],
              seq_len: int = 31) -> np.ndarray:
    """
    Parse a VICON .amc motion file.
    Returns numpy array of shape [n_frames, n_bones, 3] with joint angles (degrees).
    Random window of seq_len frames is returned.
    """
    n_bones = len(bone_names)
    bone_idx = {b: i for i, b in enumerate(bone_names)}

    frames: List[np.ndarray] = []
    current_frame: Optional[np.ndarray] = None

    try:
        with open(filepath, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith(":"):
                    continue
                parts = line.split()
                # Frame number line
                if len(parts) == 1 and parts[0].isdigit():
                    if current_frame is not None:
                        frames.append(current_frame.copy())
                    current_frame = np.zeros((n_bones, 3), dtype=np.float32)
                elif current_frame is not None and len(parts) >= 2:
                    bone = parts[0]
                    if bone not in bone_idx:
                        continue
                    idx = bone_idx[bone]
                    vals = [float(v) for v in parts[1:]]
                    # Map active DOF channels to (rx, ry, rz) slots
                    dofs = bone_dofs.get(bone, [])
                    slot_map = {"rx": 0, "ry": 1, "rz": 2,
                                "tx": 0, "ty": 1, "tz": 2}
                    for ch_i, ch in enumerate(dofs):
                        if ch in slot_map and ch_i < len(vals):
                            slot = slot_map[ch]
                            current_frame[idx, slot] = vals[ch_i]
        if current_frame is not None:
            frames.append(current_frame.copy())
    except Exception as e:
        print(f"[CMU AMC] Warning parsing {filepath}: {e}")

    if not frames:
        return np.zeros((seq_len, n_bones, 3), dtype=np.float32)

    data = np.stack(frames, axis=0)  # [T, n_bones, 3]

    # Pad to n_bones=31 if skeleton has fewer bones
    if data.shape[1] < 31:
        padded = np.zeros((data.shape[0], 31, 3), dtype=np.float32)
        padded[:, :data.shape[1], :] = data
        data = padded
    elif data.shape[1] > 31:
        data = data[:, :31, :]

    # Sample a random window of seq_len + 1 (last frame = target)
    total = data.shape[0]
    need = seq_len + 1
    if total >= need:
        start = np.random.randint(0, total - need + 1)
        return data[start: start + need]
    else:
        out = np.zeros((need, 31, 3), dtype=np.float32)
        out[:total] = data
        return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CMUMoCapDataset(Dataset):
    """
    PyTorch Dataset for CMU MoCap ASF/AMC data.
    Each sample = (x, y) where:
        x: [seq_len, 31, 3]  — input joint-angle sequence
        y: [31, 3]            — next-frame target (motion forecasting)
    Falls back to synthetic sinusoidal sequences when no AMC files are found.
    """

    CMU_BONES_EXPECTED = 31   # standard VICON full-body skeleton (excl. root → 30 + root = 31)
    SEQ_LEN = 30              # input window length

    def __init__(self,
                 data_dir: str = config.DATASET_SUBDIRS["cmu_mocap"],
                 seq_len: int = SEQ_LEN):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.samples: List[Tuple[str, str]] = []   # (asf_path, amc_path) pairs
        self.synthetic = True

        if os.path.exists(data_dir):
            self._discover_files(data_dir)

        if not self.samples:
            print(f"[CMU MoCap] No ASF/AMC pairs found in '{data_dir}'. "
                  f"Using synthetic sinusoidal fallback.")
        else:
            self.synthetic = False
            print(f"[CMU MoCap] Loaded {len(self.samples)} ASF/AMC motion sequences.")

    def _discover_files(self, root: str):
        """Find all (subject.asf, subject_XX.amc) pairs."""
        asf_files = glob.glob(os.path.join(root, "**", "*.asf"), recursive=True)
        for asf in asf_files:
            subject_dir = os.path.dirname(asf)
            amc_files = glob.glob(os.path.join(subject_dir, "*.amc"))
            for amc in amc_files:
                self.samples.append((asf, amc))

    def _synthetic_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a sinusoidal 3D joint motion sequence."""
        t = np.linspace(idx * 0.1, idx * 0.1 + 2 * np.pi, self.seq_len + 1)
        seq = np.zeros((self.seq_len + 1, 31, 3), dtype=np.float32)
        for j in range(31):
            seq[:, j, 0] = np.sin(t + j * 0.2) * 30.0
            seq[:, j, 1] = np.cos(t + j * 0.3) * 30.0
            seq[:, j, 2] = np.sin(t * 0.5 + j * 0.1) * 15.0
        x = torch.tensor(seq[:self.seq_len], dtype=torch.float32)
        y = torch.tensor(seq[self.seq_len], dtype=torch.float32)
        return x, y

    def __len__(self) -> int:
        return max(200, len(self.samples))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.synthetic:
            return self._synthetic_sample(idx)

        asf_path, amc_path = self.samples[idx % len(self.samples)]
        bone_names, bone_dofs = parse_asf(asf_path)
        data = parse_amc(amc_path, bone_names, bone_dofs, seq_len=self.seq_len)
        # data: [seq_len+1, 31, 3]
        x = torch.tensor(data[:self.seq_len], dtype=torch.float32)
        y = torch.tensor(data[self.seq_len], dtype=torch.float32)
        return x, y
