import os
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import mediapipe as mp
import config


class FacialExpressionEngine:
    """
    Pillar 1: Expressive Facial Tracking & Talking Head Engine (HDTF / Talking Face format).
    Extracts real-time facial parameters: lip stretch, mouth open ratio, eye blink, and 3D head pose.
    Utilizes MediaPipe FaceLandmarker Blendshapes (52 ARKit facial blendshapes) for exact ground truth,
    with robust geometric fallback from 33 Pose landmarks.
    """

    _face_landmarker = None
    _landmarker_initialized = False

    @classmethod
    def _init_landmarker(cls):
        """Initializes the official MediaPipe FaceLandmarker with Blendshapes."""
        if cls._landmarker_initialized:
            return
        cls._landmarker_initialized = True

        task_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "mediapipe", "face_landmarker.task")
        task_path = os.path.normpath(task_path)

        if os.path.exists(task_path):
            try:
                BaseOptions = mp.tasks.BaseOptions
                FaceLandmarker = mp.tasks.vision.FaceLandmarker
                FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
                VisionRunningMode = mp.tasks.vision.RunningMode

                options = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=task_path),
                    running_mode=VisionRunningMode.IMAGE,
                    output_face_blendshapes=True,
                    num_faces=1
                )
                cls._face_landmarker = FaceLandmarker.create_from_options(options)
                print("[FacialExpressionEngine] High-precision FaceLandmarker Blendshapes Engine active.")
            except Exception as e:
                print(f"[FacialExpressionEngine] Blendshapes initialization warning: {e}")
                cls._face_landmarker = None

    @classmethod
    def compute_expressions(cls, landmarks_33: Optional[np.ndarray], frame: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Computes normalized facial action parameters.
        Prioritizes FaceLandmarker ARKit blendshapes (jawOpen, mouthStretch, eyeBlink) from webcam frame,
        and uses Pose 33-landmarks as robust fallback.
        """
        cls._init_landmarker()

        # 1. High-precision Blendshape extraction if frame is available
        if cls._face_landmarker is not None and frame is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                res = cls._face_landmarker.detect(mp_img)

                if res and res.face_blendshapes and len(res.face_blendshapes) > 0:
                    scores = {b.category_name: b.score for b in res.face_blendshapes[0]}

                    # Exact ARKit blendshapes:
                    # jawOpen: [0.0 = completely closed, ~0.6-1.0 = wide open]
                    raw_jaw = scores.get("jawOpen", 0.0)
                    mouth_pucker = scores.get("mouthPucker", 0.0)
                    mouth_stretch = max(scores.get("mouthStretchLeft", 0.0), scores.get("mouthStretchRight", 0.0))
                    mouth_smile = max(scores.get("mouthSmileLeft", 0.0), scores.get("mouthSmileRight", 0.0))

                    # Calibrated mouth open ratio (clean cutoff when closed)
                    mouth_open_ratio = float(np.clip((raw_jaw - 0.06) / 0.55, 0.0, 1.0))
                    lip_stretch = float(np.clip(max(mouth_stretch, mouth_smile * 0.8), 0.0, 1.0))

                    # Blink
                    l_blink = scores.get("eyeBlinkLeft", 0.0)
                    r_blink = scores.get("eyeBlinkRight", 0.0)
                    avg_blink = float(np.clip((l_blink + r_blink) * 0.5, 0.0, 1.0))

                    # Head roll from landmarks_33 if available
                    roll_deg = 0.0
                    if landmarks_33 is not None and len(landmarks_33) >= 7:
                        eye_vec = landmarks_33[3][:2] - landmarks_33[6][:2]
                        roll_deg = float(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))

                    return {
                        "mouth_open_ratio": round(mouth_open_ratio, 3),
                        "lip_stretch": round(lip_stretch, 3),
                        "eye_blink_ratio": round(avg_blink, 3),
                        "head_roll_deg": round(roll_deg, 2),
                        "is_talking": mouth_open_ratio > 0.15
                    }
            except Exception:
                pass

        # 2. Geometric Fallback from 33 Pose landmarks
        if landmarks_33 is None or len(landmarks_33) < 11:
            return cls._empty_expression_dict()

        nose = landmarks_33[0][:2]
        l_eye_inner = landmarks_33[1][:2]
        l_eye_outer = landmarks_33[3][:2]
        r_eye_inner = landmarks_33[4][:2]
        r_eye_outer = landmarks_33[6][:2]
        mouth_left = landmarks_33[9][:2]
        mouth_right = landmarks_33[10][:2]

        # 1. Mouth Width / Lip Stretch Distance
        mouth_width = float(np.linalg.norm(mouth_left - mouth_right))
        eye_distance = max(1.0, float(np.linalg.norm(l_eye_outer - r_eye_outer)))

        norm_mouth_width = float(mouth_width / eye_distance)
        lip_stretch = float(np.clip((norm_mouth_width - 0.55) / 0.35, 0.0, 1.0))

        # 4. Head Pose Roll Angle (Degrees)
        eye_vector = l_eye_outer - r_eye_outer
        roll_angle_deg = float(np.degrees(np.arctan2(eye_vector[1], eye_vector[0])))

        # 2. Mouth Vertical Open Ratio (strictly down the face symmetry axis)
        face_down_unit = np.array([-eye_vector[1], eye_vector[0]]) / max(1e-5, eye_distance)
        mouth_center = (mouth_left + mouth_right) * 0.5
        nose_to_mouth_vec = mouth_center - nose
        proj_nose_mouth_dist = float(np.dot(nose_to_mouth_vec, face_down_unit))
        norm_proj_dist = float(proj_nose_mouth_dist / eye_distance)

        mouth_open_ratio = float(np.clip((norm_proj_dist - 0.37) / 0.38, 0.0, 1.0))

        # 3. Eye Blink Ratio
        l_eye_height = np.linalg.norm(l_eye_inner - l_eye_outer)
        r_eye_height = np.linalg.norm(r_eye_inner - r_eye_outer)
        avg_eye_height = (l_eye_height + r_eye_height) * 0.5
        eye_blink_ratio = float(np.clip(1.0 - (avg_eye_height / max(1.0, eye_distance * 0.25)), 0.0, 1.0))

        return {
            "mouth_open_ratio": round(mouth_open_ratio, 3),
            "lip_stretch": round(lip_stretch, 3),
            "eye_blink_ratio": round(eye_blink_ratio, 3),
            "head_roll_deg": round(roll_angle_deg, 2),
            "is_talking": mouth_open_ratio > 0.25
        }

    @staticmethod
    def _empty_expression_dict() -> Dict[str, float]:
        return {
            "mouth_open_ratio": 0.0,
            "lip_stretch": 0.0,
            "eye_blink_ratio": 0.0,
            "head_roll_deg": 0.0,
            "is_talking": False
        }
