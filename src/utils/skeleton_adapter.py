import numpy as np
from typing import Optional, Dict


class COCOTo33KeypointAdapter:
    """
    Skeleton Topology Adapter.
    Converts 17 COCO keypoints [17, 3] (x, y, visibility) into the 33-keypoint MediaPipe skeleton.
    Synthesizes missing hand/foot joints from wrist/ankle vectors.
    """

    COCO_INDEX_MAP = {
        0: 0,   # Nose -> Nose
        1: 2,   # L-Eye -> L-Eye
        2: 5,   # R-Eye -> R-Eye
        3: 7,   # L-Ear -> L-Ear
        4: 8,   # R-Ear -> R-Ear
        5: 11,  # L-Shoulder -> L-Shoulder
        6: 12,  # R-Shoulder -> R-Shoulder
        7: 13,  # L-Elbow -> L-Elbow
        8: 14,  # R-Elbow -> R-Elbow
        9: 15,  # L-Wrist -> L-Wrist
        10: 16, # R-Wrist -> R-Wrist
        11: 23, # L-Hip -> L-Hip
        12: 24, # R-Hip -> R-Hip
        13: 25, # L-Knee -> L-Knee
        14: 26, # R-Knee -> R-Knee
        15: 27, # L-Ankle -> L-Ankle
        16: 28  # R-Ankle -> R-Ankle
    }

    @classmethod
    def convert_17_to_33(cls, coco_keypoints: np.ndarray) -> np.ndarray:
        """
        Input: coco_keypoints array of shape [17, 3] or [17, 4] -> (x, y, vis) or (x, y, z, vis)
        Output: keypoints_33 array of shape [33, 4] -> (x, y, z, vis)
        """
        keypoints_33 = np.zeros((33, 4), dtype=np.float32)

        # Map direct 17 keypoints
        for coco_idx, mp_idx in cls.COCO_INDEX_MAP.items():
            if coco_idx < len(coco_keypoints):
                data = coco_keypoints[coco_idx]
                if len(data) == 3:
                    keypoints_33[mp_idx] = [data[0], data[1], 0.0, data[2]]
                else:
                    keypoints_33[mp_idx] = data[:4]

        # Synthesize Facial Landmarks 1, 3, 4, 6, 9, 10
        nose = keypoints_33[0]
        l_eye = keypoints_33[2]
        r_eye = keypoints_33[5]

        keypoints_33[1] = (nose + l_eye) * 0.5  # Left eye inner
        keypoints_33[3] = l_eye + np.array([10.0, 0.0, 0.0, 0.0])  # Left eye outer
        keypoints_33[4] = (nose + r_eye) * 0.5  # Right eye inner
        keypoints_33[6] = r_eye - np.array([10.0, 0.0, 0.0, 0.0])  # Right eye outer
        keypoints_33[9] = nose + np.array([-15.0, 35.0, 0.0, 0.0])  # Mouth left
        keypoints_33[10] = nose + np.array([15.0, 35.0, 0.0, 0.0])   # Mouth right

        # Synthesize Hand Extensions (17-22) from Wrist and Elbow vectors
        cls._synthesize_hand(keypoints_33, elbow_idx=13, wrist_idx=15, pinky_idx=17, index_idx=19, thumb_idx=21)
        cls._synthesize_hand(keypoints_33, elbow_idx=14, wrist_idx=16, pinky_idx=18, index_idx=20, thumb_idx=22)

        # Synthesize Feet Extensions (29-32) from Ankle and Knee vectors
        cls._synthesize_foot(keypoints_33, knee_idx=25, ankle_idx=27, heel_idx=29, foot_index_idx=31)
        cls._synthesize_foot(keypoints_33, knee_idx=26, ankle_idx=28, heel_idx=30, foot_index_idx=32)

        return keypoints_33

    @staticmethod
    def _synthesize_hand(kp: np.ndarray, elbow_idx: int, wrist_idx: int, pinky_idx: int, index_idx: int, thumb_idx: int):
        elbow, wrist = kp[elbow_idx], kp[wrist_idx]
        forearm_vec = wrist[:2] - elbow[:2]
        norm = np.linalg.norm(forearm_vec)
        if norm > 0:
            dir_unit = forearm_vec / norm
            perp_unit = np.array([-dir_unit[1], dir_unit[0]])
            
            # Pinky, index, thumb offsets
            kp[pinky_idx][:2] = wrist[:2] + dir_unit * 25.0 + perp_unit * 12.0
            kp[index_idx][:2] = wrist[:2] + dir_unit * 30.0 - perp_unit * 12.0
            kp[thumb_idx][:2] = wrist[:2] + dir_unit * 15.0 - perp_unit * 20.0
            kp[pinky_idx][3] = wrist[3]
            kp[index_idx][3] = wrist[3]
            kp[thumb_idx][3] = wrist[3]

    @staticmethod
    def _synthesize_foot(kp: np.ndarray, knee_idx: int, ankle_idx: int, heel_idx: int, foot_index_idx: int):
        knee, ankle = kp[knee_idx], kp[ankle_idx]
        shin_vec = ankle[:2] - knee[:2]
        norm = np.linalg.norm(shin_vec)
        if norm > 0:
            dir_unit = shin_vec / norm
            perp_unit = np.array([-dir_unit[1], dir_unit[0]])

            kp[heel_idx][:2] = ankle[:2] + dir_unit * 10.0
            kp[foot_index_idx][:2] = ankle[:2] + dir_unit * 35.0 + perp_unit * 15.0
            kp[heel_idx][3] = ankle[3]
            kp[foot_index_idx][3] = ankle[3]


class ExponentialMovingAverageSmoother:
    """
    Temporal Motion Smoother.
    Applies Exponential Moving Average (EMA) filtering across 33 keypoints to eliminate frame-to-frame jitter.
    """

    def __init__(self, alpha: float = 0.60):
        self.alpha = alpha
        self.prev_landmarks: Optional[np.ndarray] = None

    def smooth(self, current_landmarks: np.ndarray) -> np.ndarray:
        if current_landmarks is None:
            return None

        if self.prev_landmarks is None or self.prev_landmarks.shape != current_landmarks.shape:
            self.prev_landmarks = current_landmarks.copy()
            return current_landmarks

        smoothed = self.alpha * current_landmarks + (1.0 - self.alpha) * self.prev_landmarks
        self.prev_landmarks = smoothed.copy()
        return smoothed

    def reset(self):
        self.prev_landmarks = None
