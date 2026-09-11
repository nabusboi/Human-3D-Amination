import os
import cv2
import numpy as np
import urllib.request
from typing import Tuple, List, Dict, Optional
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
                 complexity: int = config.MEDIAPIPE_MODEL_COMPLEXITY,
                 min_detection_conf: float = config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
                 min_tracking_conf: float = config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE):
        self.mode = None
        self.pose_engine = None
        self.hand_engine = None

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
                    running_mode=vision.RunningMode.IMAGE,
                    min_pose_detection_confidence=min_detection_conf,
                    min_pose_presence_confidence=min_detection_conf,
                    min_tracking_confidence=min_tracking_conf,
                    num_poses=1
                )
                self.pose_engine = vision.PoseLandmarker.create_from_options(pose_opts)

            if os.path.exists(hand_path):
                hand_opts = vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=hand_path),
                    running_mode=vision.RunningMode.IMAGE,
                    min_hand_detection_confidence=min_detection_conf,
                    min_hand_presence_confidence=min_detection_conf,
                    min_tracking_confidence=min_tracking_conf,
                    num_hands=2
                )
                self.hand_engine = vision.HandLandmarker.create_from_options(hand_opts)

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
        if frame is None or self.pose_engine is None:
            return None, None, 0.0, None

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 1. Pose Detection
        pose_result = self.pose_engine.detect(mp_image)
        if not pose_result or not pose_result.pose_landmarks or len(pose_result.pose_landmarks) == 0:
            return None, None, 0.0, None

        raw_landmarks = pose_result.pose_landmarks[0]
        landmarks_array = np.zeros((33, 4), dtype=np.float32)

        for i in range(min(33, len(raw_landmarks))):
            lm = raw_landmarks[i]
            px, py = lm.x * w, lm.y * h
            vis = getattr(lm, "visibility", 1.0)
            if vis is None:
                vis = 1.0
            landmarks_array[i] = [px, py, lm.z, vis]

        # Smart extrapolation for occluded lower body & arms
        s11_vis, s12_vis = landmarks_array[11][3], landmarks_array[12][3]
        if s11_vis > 0.35 and s12_vis > 0.35:
            shoulder_width = np.linalg.norm(landmarks_array[11][:2] - landmarks_array[12][:2])
            mid_shoulder = (landmarks_array[11][:2] + landmarks_array[12][:2]) * 0.5
            torso_len = max(shoulder_width * 1.3, 120.0)

            if landmarks_array[13][3] < 0.35:
                landmarks_array[13] = [landmarks_array[11][0] - shoulder_width * 0.4, landmarks_array[11][1] + shoulder_width * 0.8, 0.0, 0.85]
            if landmarks_array[14][3] < 0.35:
                landmarks_array[14] = [landmarks_array[12][0] + shoulder_width * 0.4, landmarks_array[12][1] + shoulder_width * 0.8, 0.0, 0.85]

            if landmarks_array[15][3] < 0.35:
                landmarks_array[15] = [landmarks_array[13][0] - shoulder_width * 0.2, landmarks_array[13][1] + shoulder_width * 0.7, 0.0, 0.80]
            if landmarks_array[16][3] < 0.35:
                landmarks_array[16] = [landmarks_array[14][0] + shoulder_width * 0.2, landmarks_array[14][1] + shoulder_width * 0.7, 0.0, 0.80]

            if landmarks_array[23][3] < 0.35:
                landmarks_array[23] = [mid_shoulder[0] - shoulder_width * 0.35, mid_shoulder[1] + torso_len, 0.0, 0.85]
            if landmarks_array[24][3] < 0.35:
                landmarks_array[24] = [mid_shoulder[0] + shoulder_width * 0.35, mid_shoulder[1] + torso_len, 0.0, 0.85]

            leg_len = torso_len * 1.1
            if landmarks_array[25][3] < 0.35:
                landmarks_array[25] = [landmarks_array[23][0], landmarks_array[23][1] + leg_len, 0.0, 0.80]
            if landmarks_array[26][3] < 0.35:
                landmarks_array[26] = [landmarks_array[24][0], landmarks_array[24][1] + leg_len, 0.0, 0.80]
            if landmarks_array[27][3] < 0.35:
                landmarks_array[27] = [landmarks_array[25][0], landmarks_array[25][1] + leg_len, 0.0, 0.75]
            if landmarks_array[28][3] < 0.35:
                landmarks_array[28] = [landmarks_array[26][0], landmarks_array[26][1] + leg_len, 0.0, 0.75]

        landmark_dict = {}
        total_vis = 0.0
        for i in range(33):
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
                hand_result = self.hand_engine.detect(mp_image)
                if hand_result and hand_result.hand_landmarks and len(hand_result.hand_landmarks) > 0:
                    for idx, hand_lms in enumerate(hand_result.hand_landmarks):
                        # Determine left vs right hand
                        handedness = "Right"
                        if hand_result.handedness and idx < len(hand_result.handedness):
                            handedness = hand_result.handedness[idx][0].category_name

                        h_pts = np.zeros((21, 3), dtype=np.float32)
                        for h_i, h_lm in enumerate(hand_lms):
                            h_pts[h_i] = [h_lm.x * w, h_lm.y * h, h_lm.z]

                        if handedness == "Left":
                            hands_dict["left_hand"] = h_pts
                        else:
                            hands_dict["right_hand"] = h_pts
            except Exception as e:
                pass

        return landmarks_array, landmark_dict, avg_conf, hands_dict

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
