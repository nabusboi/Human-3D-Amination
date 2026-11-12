import os
import time
from itertools import permutations
from math import hypot
import cv2
import numpy as np
import urllib.request
from typing import Tuple, Dict, Optional
import config

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


MODEL_URL_POSE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
MODEL_PATH_POSE = os.path.join(config.MODEL_SUBDIRS.get("mediapipe", os.path.join(config.MODELS_DIR, "mediapipe")), "pose_landmarker_full.task")

MODEL_URL_HAND = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH_HAND = os.path.join(config.MODEL_SUBDIRS.get("mediapipe", os.path.join(config.MODELS_DIR, "mediapipe")), "hand_landmarker.task")


def ensure_mediapipe_models() -> Tuple[str, str]:
    """Ensures MediaPipe PoseLandmarker and HandLandmarker task model files exist locally."""
    os.makedirs(os.path.dirname(MODEL_PATH_POSE), exist_ok=True)
    if not os.path.exists(MODEL_PATH_POSE):
        print(f"[MediaPipe] Downloading PoseLandmarker model to: {MODEL_PATH_POSE}")
        try:
            urllib.request.urlretrieve(MODEL_URL_POSE, MODEL_PATH_POSE)
        except Exception as e:
            print(f"[MediaPipe Warning] Failed to download pose model: {e}")

    if not os.path.exists(MODEL_PATH_HAND):
        print(f"[MediaPipe] Downloading HandLandmarker model to: {MODEL_PATH_HAND}")
        try:
            urllib.request.urlretrieve(MODEL_URL_HAND, MODEL_PATH_HAND)
        except Exception as e:
            print(f"[MediaPipe Warning] Failed to download hand model: {e}")

    return MODEL_PATH_POSE, MODEL_PATH_HAND


class MediaPipePoseEstimator:
    """
    MediaPipe Dual Pose & 21-Keypoint Hand Tracker.
    Extracts 33 3D body landmarks + 21 3D hand landmarks per hand (Thumb, Index, Middle, Ring, Pinky).
    """

    LANDMARK_NAMES = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
        "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
        "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
        "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index"
    ]

    def __init__(self,
                 min_detection_conf: float = config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
                 min_tracking_conf: float = config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE):
        self.mode = None
        self.pose_engine = None
        self.hand_engine = None
        self.hand_recovery_engine = None
        self._last_hand_recovery = -float("inf")
        self._hand_cache = {}
        self._hand_filter = {}
        self._hand_pending = {}
        self.hands_contact = False
        self._contact_hands = None
        self._contact_seen = -float("inf")
        self._separated_since = None
        self.hand_world_landmarks = {}
        self.pose_world_landmarks = None
        self._hand_wrist_history = {}
        self._timestamp_ms = -1
        self._hand_warning_shown = False

        if not MEDIAPIPE_AVAILABLE:
            print("[MediaPipe Warning] mediapipe package not installed.")
            return

        pose_path, hand_path = ensure_mediapipe_models()

        # Initialize MediaPipe Tasks Pose & Hand Engines
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions

            if os.path.exists(pose_path):
                pose_opts = vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=pose_path),
                    running_mode=vision.RunningMode.VIDEO,
                    min_pose_detection_confidence=min_detection_conf,
                    min_pose_presence_confidence=min_detection_conf,
                    min_tracking_confidence=min_tracking_conf,
                    num_poses=1
                )
                self.pose_engine = vision.PoseLandmarker.create_from_options(pose_opts)

            if os.path.exists(hand_path):
                hand_opts = vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=hand_path),
                    running_mode=vision.RunningMode.VIDEO,
                    min_hand_detection_confidence=0.45,
                    min_hand_presence_confidence=0.45,
                    min_tracking_confidence=0.50,
                    num_hands=2
                )
                self.hand_engine = vision.HandLandmarker.create_from_options(hand_opts)
                self.hand_recovery_engine = vision.HandLandmarker.create_from_options(
                    vision.HandLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=hand_path),
                        running_mode=vision.RunningMode.IMAGE,
                        min_hand_detection_confidence=0.45,
                        min_hand_presence_confidence=0.45,
                        num_hands=2))

            self.mode = "tasks"
            print("[MediaPipe] Initialized 3D Body Pose & 21-Keypoint Hand Tracking Engine.")
            return
        except Exception as e:
            print(f"[MediaPipe Warning] Failed initializing Tasks API: {e}")

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Dict], float, Optional[Dict]]:
        """
        Processes BGR image frame and extracts 33 3D body keypoints + 21 3D hand keypoints per hand.
        Returns:
            landmarks_array: numpy array of shape [33, 4] -> [x, y, z, visibility]
            landmark_dict: dictionary mapping landmark index/name to attributes
            average_confidence: mean visibility score across keypoints
            hands_dict: dict containing 'left_hand' [21, 3] and 'right_hand' [21, 3] keypoints
        """
        self.hand_world_landmarks = {}
        self.pose_world_landmarks = None
        self.hands_contact = False
        if time.monotonic() - self._contact_seen > 0.45:
            self._contact_hands = None
        if frame is None or (self.pose_engine is None and self.hand_engine is None):
            return None, None, 0.0, None

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 1. Pose Detection
        timestamp = max(self._timestamp_ms + 1, int(time.monotonic() * 1000))
        self._timestamp_ms = timestamp
        pose_result = (self.pose_engine.detect_for_video(mp_image, timestamp)
                       if self.pose_engine is not None else None)
        landmarks_array = None
        if pose_result and pose_result.pose_landmarks:
            raw_landmarks = pose_result.pose_landmarks[0]
            landmarks_array = np.zeros((33, 4), dtype=np.float32)
            for i, lm in enumerate(raw_landmarks[:33]):
                vis = getattr(lm, "visibility", 1.0)
                landmarks_array[i] = [lm.x * w, lm.y * h, lm.z,
                                      vis if vis is not None else 1.0]
            if pose_result.pose_world_landmarks:
                metric = np.asarray([[lm.x, lm.y, lm.z]
                                     for lm in pose_result.pose_world_landmarks[0]], dtype=np.float32)
                if metric.shape == (33, 3) and np.isfinite(metric).all():
                    self.pose_world_landmarks = metric

        # Preserve actual visibility. The native rig uses its idle pose for
        # occluded joints; invented high-confidence positions bend it incorrectly.

        landmark_dict = {} if landmarks_array is not None else None
        total_vis = 0.0
        for i in range(33 if landmarks_array is not None else 0):
            px, py, z, vis = landmarks_array[i]
            name = self.LANDMARK_NAMES[i] if i < len(self.LANDMARK_NAMES) else f"lm_{i}"
            landmark_dict[i] = {
                "name": name, "x": px, "y": py, "z": z, "visibility": vis,
                "norm_x": px / w, "norm_y": py / h
            }
            total_vis += vis
        avg_conf = total_vis / 33.0

        # 2. 21-Keypoint Hand Detection per Hand
        hands_dict = {"left_hand": None, "right_hand": None}
        if self.hand_engine is not None:
            try:
                hand_result = self.hand_engine.detect_for_video(mp_image, timestamp)
                # VIDEO tracking may keep following one palm when two overlap.
                # Periodically run fresh detection, without duplicating a palm.
                now = time.monotonic()
                if (self.hand_recovery_engine is not None
                        and len(hand_result.hand_landmarks) < 2
                        and self._contact_hands is None
                        and now - self._last_hand_recovery >= 0.35):
                    self._last_hand_recovery = now
                    recovered = self.hand_recovery_engine.detect(mp_image)
                    if len(recovered.hand_landmarks) > len(hand_result.hand_landmarks):
                        hand_result = recovered
                if hand_result and hand_result.hand_landmarks:
                    hands_dict = self._assign_hands(hand_result, landmarks_array, w, h)
                elif self._contact_hands is not None and now - self._contact_seen < 0.35:
                    self.hands_contact = True
                    for key, (points, world) in self._contact_hands.items():
                        hands_dict[key] = points.copy()
                        if world is not None:
                            self.hand_world_landmarks[key] = world.copy()
                for key in hands_dict:
                    if hands_dict[key] is not None and not self.hands_contact:
                        self._hand_cache[key] = (
                            hands_dict[key].copy(), self.hand_world_landmarks.get(key), now)
                    elif hands_dict[key] is None and key in self._hand_cache:
                        points, world, seen = self._hand_cache[key]
                        # Brief occlusion hold, not a new detection. Never refresh
                        # its timestamp from held data or invent the other hand.
                        if now - seen < 0.35:
                            hands_dict[key] = points.copy()
                            if world is not None:
                                self.hand_world_landmarks[key] = world
            except Exception as e:
                if not self._hand_warning_shown:
                    print(f"[MediaPipe Warning] Hand tracking failed: {e}")
                    self._hand_warning_shown = True

        return landmarks_array, landmark_dict, avg_conf, hands_dict

    def _assign_hands(self, result, pose, width, height):
        """Associate hands with body wrists, including mirrored camera input.

        Handedness labels alone can disagree with the pose skeleton. The
        one-to-one match prevents one raised hand from driving the other arm.
        Metric hand geometry is kept separately from image-space wrist anchors.
        """
        keys = ("left_hand", "right_hand")
        hands = dict.fromkeys(keys)
        detections = []
        for i, points in enumerate(result.hand_landmarks[:2]):
            image_points = np.asarray(
                [[p.x * width, p.y * height, p.z] for p in points], dtype=np.float32)
            if image_points.shape != (21, 3) or not np.isfinite(image_points).all():
                continue
            label = None
            if i < len(result.handedness) and result.handedness[i]:
                label = result.handedness[i][0].category_name.lower() + "_hand"
            world = None
            if i < len(result.hand_world_landmarks):
                world = np.asarray([[p.x, p.y, p.z]
                                    for p in result.hand_world_landmarks[i]], dtype=np.float32)
                if world.shape != (21, 3) or not np.isfinite(world).all():
                    world = None
            detections.append((image_points, world, label))
        if not detections:
            return hands

        now = time.monotonic()
        guarded = self._guard_hand_contact(detections, pose, now)
        if guarded is not None:
            return guarded
        costs = np.zeros((len(detections), 2))
        for row, (points, _, label) in enumerate(detections):
            for side, wrist in enumerate((15, 16)):
                anchor = None
                if (pose is not None and pose[wrist, 3] >= 0.65
                        and np.isfinite(pose[wrist, :2]).all()
                        and 0 <= pose[wrist, 0] < width and 0 <= pose[wrist, 1] < height):
                    anchor = pose[wrist, :2]
                elif keys[side] in self._hand_wrist_history:
                    previous, seen = self._hand_wrist_history[keys[side]]
                    if now - seen < 0.4:
                        anchor = previous
                label_penalty = 0.0 if label == keys[side] else 0.04
                if anchor is not None:
                    distance = np.linalg.norm(points[0, :2] - anchor) / hypot(width, height)
                    costs[row, side] = distance + label_penalty
                else:
                    costs[row, side] = 0.20 + label_penalty
                previous = self._hand_wrist_history.get(keys[side])
                if previous is not None and now - previous[1] < 0.4:
                    costs[row, side] += 0.5 * np.linalg.norm(
                        points[0, :2] - previous[0]) / hypot(width, height)
        assignment = min(permutations(range(2), len(detections)),
                         key=lambda order: sum(costs[i, side]
                                               for i, side in enumerate(order)))
        for (points, world, _), side in zip(detections, assignment):
            key = keys[side]
            filtered = self._stabilize_hand(key, points, world, now)
            if filtered is None:
                continue
            points, world = filtered
            hands[key] = points
            if world is not None:
                self.hand_world_landmarks[key] = world
            self._hand_wrist_history[key] = (points[0, :2].copy(), now)
        return hands

    def _stabilize_hand(self, key, points, world, now):
        """Filter measured hands before both wrist IK and finger retargeting.

        Compare against the accepted pose, so slow intentional movement can
        accumulate past the deadband. Ignore depth-only noise while the image
        landmarks are still; require repeated evidence for large pose jumps.
        """
        if world is not None:
            world = world - world[0]
        previous = self._hand_filter.get(key)
        if previous is not None and now - previous[2] < 0.45:
            old, old_world, seen = previous
            palm = max(12.0, float(np.linalg.norm(old[9, :2] - old[0, :2])))
            movement = float(np.max(np.linalg.norm(points[:, :2] - old[:, :2], axis=1)))
            if movement < max(1.5, palm * 0.025):
                self._hand_pending.pop(key, None)
                self._hand_filter[key] = (old, old_world, now)
                return old.copy(), None if old_world is None else old_world.copy()
            depth_jump = (old_world is not None and world is not None
                          and np.max(np.linalg.norm(world - old_world, axis=1)) > 0.08)
            if movement > palm * 0.80 or depth_jump:
                pending = self._hand_pending.get(key)
                consistent = (pending is not None
                              and np.max(np.linalg.norm(points[:, :2] - pending[0][:, :2], axis=1)) < palm * 0.25)
                if consistent and world is not None and pending[1] is not None:
                    consistent = np.max(np.linalg.norm(world - pending[1], axis=1)) < 0.035
                if not consistent:
                    self._hand_pending[key] = (points.copy(), world, now)
                    return None
                if now - pending[2] < 0.15:
                    return None
            self._hand_pending.pop(key, None)
            dt = min(0.15, max(0.001, now - seen))
            tau = 0.045 if movement > palm * 0.15 else 0.12
            alpha = 1.0 - np.exp(-dt / tau)
            points = old + (points - old) * alpha
            if old_world is not None and world is not None:
                world = old_world + (world - old_world) * alpha
        else:
            self._hand_pending.pop(key, None)
        self._hand_filter[key] = (points.copy(), None if world is None else world.copy(), now)
        return points, world

    def _guard_hand_contact(self, detections, pose, now):
        """Overlapping palms do not provide reliable side or finger depth.

        Keep separate, previously observed hand shapes through contact.
        A single detected palm is never copied onto the other hand.
        """
        keys = ("left_hand", "right_hand")
        palms = [entry[0] for entry in detections]
        size = max(12.0, float(np.median([
            np.linalg.norm(p[9, :2] - p[0, :2]) for p in palms])))
        center = np.mean([p[0, :2] for p in palms], axis=0)
        close = False
        if len(palms) == 2:
            wrist_gap = np.linalg.norm(palms[0][0, :2] - palms[1][0, :2])
            palm_gap = np.linalg.norm(
                palms[0][[0, 5, 9, 13, 17], :2].mean(axis=0)
                - palms[1][[0, 5, 9, 13, 17], :2].mean(axis=0))
            threshold = 1.60 if self._contact_hands is not None else 1.10
            close = wrist_gap < threshold * size and palm_gap < threshold * size
        elif self._contact_hands is not None:
            previous_center = np.mean(
                [entry[0][0, :2] for entry in self._contact_hands.values()], axis=0)
            close = np.linalg.norm(center - previous_center) < size * 1.5
            if (pose is not None and np.min(pose[[15, 16], 3]) >= 0.75
                    and np.linalg.norm(pose[15, :2] - pose[16, :2]) > size * 2):
                close = False
        elif pose is not None and np.min(pose[[15, 16], 3]) >= 0.65:
            # One palm can vanish just as the wrists meet.
            wrists = pose[[15, 16], :2]
            close = (np.linalg.norm(wrists[0] - wrists[1]) < size * 0.85
                     and np.linalg.norm(center - wrists.mean(axis=0)) < size)

        if close:
            self._separated_since = None
            self._contact_seen = now
            if self._contact_hands is None:
                recent = {key: self._hand_cache[key] for key in keys
                          if key in self._hand_cache
                          and now - self._hand_cache[key][2] < 0.45}
                if len(recent) == 2:
                    self._contact_hands = {
                        key: (points.copy(), None if world is None else world.copy())
                        for key, (points, world, _) in recent.items()}
            self.hands_contact = True
        elif self._contact_hands is not None:
            if self._separated_since is None:
                self._separated_since = now
            # Require sustained separation, not a single noisy detection.
            if now - self._separated_since >= 0.25:
                self._contact_hands = None
            else:
                self.hands_contact = True

        if not self.hands_contact:
            return None
        if self._contact_hands is None:
            # No reliable pair before contact: do not invent a prayer pose.
            return dict.fromkeys(keys)
        if len(palms) == 2 and close:
            old_center = np.mean(
                [entry[0][0, :2] for entry in self._contact_hands.values()], axis=0)
            shift = center - old_center
            length = np.linalg.norm(shift)
            shift *= min(0.25, size * 0.12 / max(length, 1e-6))
            for points, _ in self._contact_hands.values():
                points[:, :2] += shift
        held = {}
        for key, (points, world) in self._contact_hands.items():
            held[key] = points.copy()
            if world is not None:
                self.hand_world_landmarks[key] = world.copy()
            self._hand_wrist_history[key] = (points[0, :2].copy(), now)
        return held

    def close(self):
        for name in ("pose_engine", "hand_engine", "hand_recovery_engine"):
            engine = getattr(self, name, None)
            if engine is not None:
                engine.close()
                setattr(self, name, None)

    def validate_precheck(self, landmarks_array: Optional[np.ndarray]) -> Tuple[bool, str, float, float]:
        """Pre-recording check: Verifies face and shoulders are detected."""
        if landmarks_array is None:
            return False, "No human subject detected", 0.0, 0.0

        face_indices = config.REQUIRED_FACE_LANDMARKS
        body_indices = [11, 12]

        face_vis = landmarks_array[face_indices, 3]
        body_vis = landmarks_array[body_indices, 3]

        avg_face_conf = float(np.mean(face_vis))
        avg_body_conf = float(np.mean(body_vis))

        if avg_face_conf < 0.35:
            return False, f"Face unclear ({int(avg_face_conf * 100)}%)", avg_face_conf, avg_body_conf

        if avg_body_conf < 0.35:
            return False, f"Shoulders obstructed ({int(avg_body_conf * 100)}%)", avg_face_conf, avg_body_conf

        return True, "READY TO RECORD (3D Avatar Active)", avg_face_conf, avg_body_conf

    @staticmethod
    def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculates 2D joint angle between 3 keypoints in degrees (p2 as vertex)."""
        v1 = p1[:2] - p2[:2]
        v2 = p3[:2] - p2[:2]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cosine_angle = np.dot(v1, v2) / (norm1 * norm2)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine_angle)))
