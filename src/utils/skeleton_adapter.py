import numpy as np
from typing import Optional


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
