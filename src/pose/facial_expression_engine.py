import os
from typing import Dict, Optional

import cv2
import mediapipe as mp
import numpy as np


class FacialExpressionEngine:
    """MediaPipe face tracking and compact landmark inputs for the HDTF model."""

    _face_landmarker = None
    _landmarker_initialized = False
    _MODEL_LANDMARK_IDS = (1, 33, 133, 159, 145, 362, 263, 386, 374, 61, 291)

    @classmethod
    def close(cls):
        if cls._face_landmarker is not None:
            cls._face_landmarker.close()
            cls._face_landmarker = None
        cls._landmarker_initialized = False

    @classmethod
    def _init_landmarker(cls):
        if cls._landmarker_initialized:
            return
        cls._landmarker_initialized = True
        task_path = os.path.normpath(os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "mediapipe", "face_landmarker.task"
        ))
        if not os.path.exists(task_path):
            print(f"[FacialExpressionEngine] FaceLandmarker model not found: {task_path}")
            return
        try:
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=task_path),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1,
            )
            cls._face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            print("[FacialExpressionEngine] FaceLandmarker tracking active.")
        except Exception as exc:
            print(f"[FacialExpressionEngine] Initialization failed: {exc}")

    @classmethod
    def track_face(cls, frame: Optional[np.ndarray]) -> Optional[Dict]:
        """Return face metrics, render anchors in frame pixels, and [11,3] model input."""
        cls._init_landmarker()
        if cls._face_landmarker is None or frame is None:
            return None
        try:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = cls._face_landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            )
            if not result or not result.face_landmarks:
                return None

            landmarks = result.face_landmarks[0]
            if len(landmarks) <= 454:
                return None

            def point(index):
                p = landmarks[index]
                return (int(np.clip(p.x * w, 0, w - 1)), int(np.clip(p.y * h, 0, h - 1)))

            def midpoint(a, b):
                pa, pb = np.asarray(point(a)), np.asarray(point(b))
                return tuple(np.rint((pa + pb) * 0.5).astype(int))

            scores = {}
            if result.face_blendshapes:
                scores = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
            jaw = scores.get("jawOpen", 0.0)
            mouth_open = float(np.clip((jaw - 0.04) / 0.50, 0.0, 1.0))
            lip_stretch = float(np.clip(max(
                scores.get("mouthStretchLeft", 0.0),
                scores.get("mouthStretchRight", 0.0),
                0.8 * max(scores.get("mouthSmileLeft", 0.0), scores.get("mouthSmileRight", 0.0)),
            ), 0.0, 1.0))
            blink = float(np.clip((scores.get("eyeBlinkLeft", 0.0) + scores.get("eyeBlinkRight", 0.0)) * 0.5, 0.0, 1.0))

            left_eye = midpoint(33, 133)
            right_eye = midpoint(362, 263)
            iris_centers = [left_eye, right_eye]
            if len(landmarks) > 473:
                eye_points = (left_eye, right_eye)
                candidates = (point(468), point(473))
                assigned = [None, None]
                for candidate in candidates:
                    nearest = int(np.argmin([np.linalg.norm(np.asarray(candidate) - np.asarray(e)) for e in eye_points]))
                    assigned[nearest] = candidate
                iris_centers = [assigned[i] if assigned[i] is not None else eye_points[i] for i in range(2)]
            eye_widths = (
                max(4.0, float(np.linalg.norm(np.asarray(point(33)) - np.asarray(point(133))))),
                max(4.0, float(np.linalg.norm(np.asarray(point(362)) - np.asarray(point(263))))),
            )
            gaze = []
            for eye, iris, width in zip((left_eye, right_eye), iris_centers, eye_widths):
                dx = (iris[0] - eye[0]) / max(1.0, width * 0.42)
                dy = (eye[1] - iris[1]) / max(1.0, width * 0.22)
                gaze.append((float(np.clip(dx, -1.0, 1.0)), float(np.clip(dy, -1.0, 1.0))))
            eye_delta = np.asarray(right_eye, dtype=np.float32) - np.asarray(left_eye, dtype=np.float32)
            roll = float(np.degrees(np.arctan2(eye_delta[1], eye_delta[0])))
            face_points = np.asarray([point(i) for i in (10, 152, 234, 454)], dtype=np.float32)
            x0, y0 = face_points.min(axis=0)
            x1, y1 = face_points.max(axis=0)

            model_input = np.asarray([
                [landmarks[i].x, landmarks[i].y, 1.0] for i in cls._MODEL_LANDMARK_IDS
            ], dtype=np.float32)
            return {
                "expressions": {
                    "mouth_open_ratio": mouth_open,
                    "lip_stretch": lip_stretch,
                    "eye_blink_ratio": blink,
                    "head_roll_deg": roll,
                    "is_talking": mouth_open > 0.12,
                },
                "blendshapes": scores,
                "model_landmarks": model_input,
                "center": ((x0 + x1) * 0.5, (y0 + y1) * 0.5),
                "face_width": max(1.0, float(x1 - x0)),
                "face_height": max(1.0, float(y1 - y0)),
                "left_eye": left_eye,
                "right_eye": right_eye,
                "left_eye_width": max(4, int(np.linalg.norm(np.asarray(point(33)) - np.asarray(point(133))) * 0.65)),
                "right_eye_width": max(4, int(np.linalg.norm(np.asarray(point(362)) - np.asarray(point(263))) * 0.65)),
                "mouth_center": midpoint(13, 14),
                "mouth_width": max(8, int(np.linalg.norm(np.asarray(point(61)) - np.asarray(point(291))) * 0.60)),
                "mouth_height": max(2, int(np.linalg.norm(np.asarray(point(13)) - np.asarray(point(14))))),
                "eye_centers": (left_eye, right_eye),
                "iris_centers": tuple(iris_centers),
                "eye_gaze": tuple(gaze),
                "facial_pose_matrix": (
                    np.asarray(result.facial_transformation_matrixes[0], dtype=np.float32).tolist()
                    if result.facial_transformation_matrixes else None
                ),
            }
        except Exception as exc:
            print(f"[FacialExpressionEngine] Face tracking failed: {exc}")
            return None

    @classmethod
    def compute_expressions(cls, landmarks_33: Optional[np.ndarray], frame: Optional[np.ndarray] = None,
                            face_tracking: Optional[Dict] = None) -> Dict[str, float]:
        if face_tracking and face_tracking.get("expressions"):
            return dict(face_tracking["expressions"])
        if frame is not None:
            tracked = cls.track_face(frame)
            if tracked:
                return dict(tracked["expressions"])
        if landmarks_33 is None or len(landmarks_33) < 11:
            return cls._empty_expression_dict()

        nose = landmarks_33[0][:2]
        left_outer, right_outer = landmarks_33[3][:2], landmarks_33[6][:2]
        eye_vec = right_outer - left_outer
        eye_distance = max(1.0, float(np.linalg.norm(eye_vec)))
        roll = float(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))
        mouth_left, mouth_right = landmarks_33[9][:2], landmarks_33[10][:2]
        mouth_width = float(np.linalg.norm(mouth_left - mouth_right))
        lip_stretch = float(np.clip((mouth_width / eye_distance - 0.55) / 0.35, 0.0, 1.0))
        down = np.array([-eye_vec[1], eye_vec[0]], dtype=np.float32) / eye_distance
        mouth_center = (mouth_left + mouth_right) * 0.5
        vertical = float(np.dot(mouth_center - nose, down) / eye_distance)
        mouth_open = float(np.clip((vertical - 0.37) / 0.38, 0.0, 1.0))
        return {
            "mouth_open_ratio": mouth_open, "lip_stretch": lip_stretch,
            "eye_blink_ratio": 0.0, "head_roll_deg": roll,
            "is_talking": mouth_open > 0.25,
        }

    @staticmethod
    def _empty_expression_dict() -> Dict[str, float]:
        return {"mouth_open_ratio": 0.0, "lip_stretch": 0.0,
                "eye_blink_ratio": 0.0, "head_roll_deg": 0.0, "is_talking": False}
