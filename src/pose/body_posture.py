"""Conservative posture estimates from visible joints and metric pose geometry."""

import time

import numpy as np


class BodyPostureTracker:
    ARM_BONES = (
        ("LeftArm", 11, 13), ("LeftForeArm", 13, 15),
        ("RightArm", 12, 14), ("RightForeArm", 14, 16),
    )
    LEG_BONES = (
        ("LeftUpLeg", 23, 25), ("LeftLeg", 25, 27), ("LeftFoot", 27, 31),
        ("RightUpLeg", 24, 26), ("RightLeg", 26, 28), ("RightFoot", 28, 32),
    )

    def __init__(self):
        self.state = "unknown"
        self.coverage = "portrait"
        self._pending_state = self._pending_coverage = None
        self._state_since = self._coverage_since = 0.0
        self._last_time = None

    @staticmethod
    def _unit(vector):
        length = np.linalg.norm(vector)
        return vector / length if length > 1e-6 else None

    @classmethod
    def _angle(cls, a, joint, b):
        first, second = cls._unit(a - joint), cls._unit(b - joint)
        if first is None or second is None:
            return None
        return float(np.degrees(np.arccos(np.clip(np.dot(first, second), -1, 1))))

    def update(self, image_points, world_points, width, height, now=None):
        now = time.monotonic() if now is None else float(now)
        if self._last_time is not None and now - self._last_time > 0.6:
            # A paused camera must not count as sustained evidence.
            self._pending_state = self._pending_coverage = None
        self._last_time = now
        image = np.asarray(image_points) if image_points is not None else np.empty(0)
        world = np.asarray(world_points) if world_points is not None else np.empty(0)
        visible = np.zeros(33, dtype=bool)
        if image.shape == (33, 4):
            visible = (np.isfinite(image).all(axis=1) & (image[:, 3] >= 0.60)
                       & (image[:, 0] >= 0) & (image[:, 0] < width)
                       & (image[:, 1] >= 0) & (image[:, 1] < height))
        coverage = "portrait"
        if np.all(visible[[11, 12, 23, 24]]):
            coverage = "waist"
            if np.all(visible[[25, 26]]):
                coverage = "knees"
                if np.all(visible[[27, 28]]):
                    coverage = "full"
        if coverage != self._pending_coverage:
            self._pending_coverage, self._coverage_since = coverage, now
        if now - self._coverage_since >= 0.5:
            self.coverage = coverage

        directions, arm_directions, torso, across = {}, {}, None, None
        hip_angles, knee_angles, thigh_down = [], [], []
        candidate = None
        confidence = 0.0
        metric = world.shape == (33, 3) and np.isfinite(world).all()
        if metric:
            # MediaPipe metres (right, down, away) -> Panda (right, away, up).
            positions = world[:, [0, 2, 1]].copy()
            positions[:, 2] *= -1
            for name, start, end in self.ARM_BONES:
                if visible[start] and visible[end]:
                    vector = positions[end] - positions[start]
                    if 0.025 <= np.linalg.norm(vector) <= 0.65:
                        arm_directions[name] = self._unit(vector)
            for name, start, end in self.LEG_BONES:
                if visible[start] and visible[end]:
                    vector = positions[end] - positions[start]
                    # Reject collapsed / implausible segments before retargeting.
                    if 0.035 <= np.linalg.norm(vector) <= 0.85:
                        directions[name] = self._unit(vector)
            if np.all(visible[[11, 12, 23, 24]]):
                hips = (positions[23] + positions[24]) * 0.5
                shoulders = (positions[11] + positions[12]) * 0.5
                torso = self._unit(shoulders - hips)
                across = self._unit(positions[23] - positions[24])
                if torso is not None:
                    for shoulder, hip, knee, ankle in ((11, 23, 25, 27), (12, 24, 26, 28)):
                        if visible[knee]:
                            angle = self._angle(positions[shoulder], positions[hip], positions[knee])
                            thigh = self._unit(positions[knee] - positions[hip])
                            if angle is not None and thigh is not None:
                                hip_angles.append(angle)
                                thigh_down.append(float(np.dot(thigh, -torso)))
                        if np.all(visible[[hip, knee, ankle]]):
                            angle = self._angle(positions[hip], positions[knee], positions[ankle])
                            if angle is not None:
                                knee_angles.append(angle)
                    # Both thighs must agree. Raising one knee is not sitting.
                    if len(hip_angles) == 2 and all(
                            name in directions for name in ("LeftUpLeg", "RightUpLeg")):
                        confidence = float(np.min(image[[11, 12, 23, 24, 25, 26], 3]))
                        knee_bent = not knee_angles or max(knee_angles) < 145
                        knee_straight = not knee_angles or min(knee_angles) > 150
                        if max(hip_angles) < 130 and max(thigh_down) < 0.60 and knee_bent:
                            candidate = "sitting"
                        elif min(hip_angles) > 153 and min(thigh_down) > 0.86 and knee_straight:
                            candidate = "standing"
        if candidate != self._pending_state:
            self._pending_state, self._state_since = candidate, now
        if candidate is not None and now - self._state_since >= 0.35:
            self.state = candidate
        tracked = candidate is not None and candidate == self.state
        return {
            "state": self.state,
            "tracked": tracked,
            "confidence": confidence if tracked else 0.0,
            "coverage": self.coverage,
            "arm_directions": arm_directions,
            "leg_directions": directions,
            "torso_direction": torso,
            "hip_direction": across,
        }
