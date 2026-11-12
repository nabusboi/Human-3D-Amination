import numpy as np
from typing import Dict, Optional


class MoCap3DRig:
    """
    Pillar 2: Full-Body Motion Capture (MoCap) 3D Rigging Engine (CMU MoCap / AMASS format).
    Calculates joint angles used in recording telemetry.
    """

    @classmethod
    def compute_mocap_rig(cls, landmarks_33: Optional[np.ndarray]) -> Dict:
        """
        Compute elbow and knee angles from the tracked landmarks.
        """
        if landmarks_33 is None:
            return {"joint_angles": {}}

        # 1. Calculate 3D Joint Angles
        l_elbow_angle = cls._calculate_3d_angle(landmarks_33[11], landmarks_33[13], landmarks_33[15])
        r_elbow_angle = cls._calculate_3d_angle(landmarks_33[12], landmarks_33[14], landmarks_33[16])
        l_knee_angle = cls._calculate_3d_angle(landmarks_33[23], landmarks_33[25], landmarks_33[27])
        r_knee_angle = cls._calculate_3d_angle(landmarks_33[24], landmarks_33[26], landmarks_33[28])

        return {
            "joint_angles": {
                "left_elbow": round(l_elbow_angle, 1),
                "right_elbow": round(r_elbow_angle, 1),
                "left_knee": round(l_knee_angle, 1),
                "right_knee": round(r_knee_angle, 1)
            }
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
