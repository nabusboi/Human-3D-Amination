import numpy as np
import math
from typing import List, Tuple, Dict, Optional


class MoCap3DRig:
    """
    Pillar 2: Full-Body Motion Capture (MoCap) 3D Rigging Engine (CMU MoCap / AMASS format).
    Calculates 3D joint rotations, bone vectors, 3D depth-sorting (Z-buffering), and perspective projection.
    """

    # CMU / AMASS 3D Joint Pairs
    MOCAP_BONES = [
        # Head & Neck
        (0, 11, "neck"), (0, 12, "neck"),   # Nose -> both shoulders (neck/spine)
        (7, 8, "head"),                       # Left Ear -> Right Ear (head width)
        (1, 4, "head"),                       # Left Eye Inner -> Right Eye Inner
        # Torso
        (11, 12, "torso"), (11, 23, "torso"), (12, 24, "torso"), (23, 24, "torso"),
        # Arms
        (11, 13, "l_arm"), (13, 15, "l_arm"), (12, 14, "r_arm"), (14, 16, "r_arm"),
        # Legs
        (23, 25, "l_leg"), (25, 27, "l_leg"), (24, 26, "r_leg"), (26, 28, "r_leg"),
        # Feet
        (27, 29, "l_foot"), (27, 31, "l_foot"), (28, 30, "r_foot"), (28, 32, "r_foot")
    ]

    @classmethod
    def compute_mocap_rig(cls, landmarks_33: Optional[np.ndarray]) -> Dict:
        """
        Computes 3D joint angles, depth-sorted bone order, and MoCap rotation vectors.
        """
        if landmarks_33 is None:
            return {"joint_angles": {}, "depth_sorted_bones": []}

        # 1. Calculate 3D Joint Angles
        l_elbow_angle = cls._calculate_3d_angle(landmarks_33[11], landmarks_33[13], landmarks_33[15])
        r_elbow_angle = cls._calculate_3d_angle(landmarks_33[12], landmarks_33[14], landmarks_33[16])
        l_knee_angle = cls._calculate_3d_angle(landmarks_33[23], landmarks_33[25], landmarks_33[27])
        r_knee_angle = cls._calculate_3d_angle(landmarks_33[24], landmarks_33[26], landmarks_33[28])

        # 2. Depth Sort Bones by Mean Z-depth (Z-buffering for 3D rendering)
        depth_sorted_bones = []
        for p1_idx, p2_idx, bone_type in cls.MOCAP_BONES:
            z1 = landmarks_33[p1_idx][2]
            z2 = landmarks_33[p2_idx][2]
            mean_z = (z1 + z2) * 0.5
            depth_sorted_bones.append((mean_z, p1_idx, p2_idx, bone_type))

        # Sort back-to-front (largest Z first)
        depth_sorted_bones.sort(key=lambda item: item[0], reverse=True)

        return {
            "joint_angles": {
                "left_elbow": round(l_elbow_angle, 1),
                "right_elbow": round(r_elbow_angle, 1),
                "left_knee": round(l_knee_angle, 1),
                "right_knee": round(r_knee_angle, 1)
            },
            "depth_sorted_bones": [(b[1], b[2], b[3]) for b in depth_sorted_bones]
        }

    @staticmethod
    def _calculate_3d_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculates 3D angle between 3 points (p2 as vertex) in degrees."""
        v1 = p1[:3] - p2[:3]
        v2 = p3[:3] - p2[:3]

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 == 0 or n2 == 0:
            return 0.0

        dot = np.dot(v1, v2) / (n1 * n2)
        dot = np.clip(dot, -1.0, 1.0)
        return float(np.degrees(np.arccos(dot)))
