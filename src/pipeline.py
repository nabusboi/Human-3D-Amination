import os
import torch
import cv2
import numpy as np
from typing import Tuple, Dict
import config
from src.detector.yolo_detector import YOLOHumanDetector
from src.pose.mediapipe_pose import MediaPipePoseEstimator
from src.pose.facial_expression_engine import FacialExpressionEngine
from src.pose.body_posture import BodyPostureTracker
from src.utils.mocap_rig import MoCap3DRig
from src.utils.skeleton_adapter import ExponentialMovingAverageSmoother
from src.animator.avatar_3d_animator import Avatar3DAnimator
from src.recorder.motion_recorder import MotionRecorder, RecordingState
from src.models.unified_multitask_transformer import UnifiedMultiTaskTransformer


class MovementDetectorPipeline:
    """
    Unified 3D Avatar Movement Detector Pipeline.
    Connects: Webcam Stream -> YOLO Person Detector -> MediaPipe 33 Landmark Pose ->
              Unified QLoRA Multi-Task Model -> HDTF Facial Engine -> MoCap 3D Rig ->
              EMA Smoother -> 3D Avatar Animator -> Motion Recorder.
    """

    def __init__(self):
        self.detector = YOLOHumanDetector()
        self.pose_estimator = MediaPipePoseEstimator()
        self.posture_tracker = BodyPostureTracker()
        self.smoother = ExponentialMovingAverageSmoother(alpha=0.60)
        self.animator = Avatar3DAnimator(canvas_width=config.FRAME_WIDTH, canvas_height=config.FRAME_HEIGHT)
        self.recorder = MotionRecorder()
        self._face_track_ema = None
        self._frame_index = 0
        self._cached_coco = None

        # Load Trained Unified Multi-Task QLoRA Transformer Model
        device_preference = os.getenv("AVATAR_DEVICE", "auto").strip().lower()
        cuda_free_bytes = 0
        if torch.cuda.is_available():
            try:
                cuda_free_bytes, _ = torch.cuda.mem_get_info()
            except Exception:
                pass
        enough_cuda_memory = cuda_free_bytes >= 1024 ** 3
        use_cuda = torch.cuda.is_available() and device_preference != "cpu" and (
            device_preference == "cuda" or enough_cuda_memory
        )
        if device_preference not in ("auto", "cpu", "cuda"):
            print(f"[Pipeline Note] Unknown AVATAR_DEVICE={device_preference!r}; using automatic device selection.")
        self.device = torch.device("cuda" if use_cuda else "cpu")
        if torch.cuda.is_available() and device_preference == "auto" and not enough_cuda_memory:
            print(f"[Pipeline] GPU has {cuda_free_bytes / (1024 ** 2):.0f} MiB free; using CPU to avoid exhausting VRAM.")
        self.model = None
        self._model_inference_warning_shown = False
        self.model_path = os.path.join(config.MODEL_SUBDIRS.get("qlora_pose", os.path.join(config.MODELS_DIR, "qlora_pose")), "unified_multitask_qlora.pt")

        if os.path.exists(self.model_path):
            try:
                self.model = UnifiedMultiTaskTransformer(
                    embed_dim=256, num_heads=8, num_layers=4,
                    r=config.LORA_R, alpha=config.LORA_ALPHA
                ).to(self.device)
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.eval()
                print(f"[Pipeline] Successfully loaded Unified Multi-Task QLoRA Model from: {self.model_path}")
            except Exception as e:
                self.model = None
                print(f"[Pipeline Warning] Failed to load trained model: {e}")
        else:
            print(f"[Pipeline Note] No trained model checkpoint found at {self.model_path}. Using real-time MediaPipe estimation.")



    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Processes camera input frame and generates live 3D Avatar Animation preview.
        """
        if frame is None:
            blank = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
            return blank, blank, {}

        if frame.shape[1] != config.FRAME_WIDTH or frame.shape[0] != config.FRAME_HEIGHT:
            scale = min(config.FRAME_WIDTH / frame.shape[1],
                        config.FRAME_HEIGHT / frame.shape[0])
            resized = cv2.resize(frame, (round(frame.shape[1] * scale),
                                        round(frame.shape[0] * scale)),
                                 interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=frame.dtype)
            x = (config.FRAME_WIDTH - resized.shape[1]) // 2
            y = (config.FRAME_HEIGHT - resized.shape[0]) // 2
            canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
            frame = canvas

        if config.MIRROR_WEBCAM:
            frame = cv2.flip(frame, 1)

        # 1. YOLO Human Subject Bounding Box
        bbox, yolo_conf, _ = self.detector.detect_person(frame)

        # 2. MediaPipe 33 Body Keypoints + 21 Hand Keypoints
        landmarks_33, lm_dict, pose_conf, hands_dict = self.pose_estimator.process_frame(frame)
        body_tracking = self.posture_tracker.update(
            landmarks_33, self.pose_estimator.pose_world_landmarks,
            frame.shape[1], frame.shape[0])
        body_tracking["hands_contact"] = self.pose_estimator.hands_contact

        # 3. Motion Smoothing
        smoothed_lm = self.smoother.smooth(landmarks_33) if landmarks_33 is not None else None

        # FaceLandmarker gives dense facial geometry and ARKit expression signals.
        face_tracking = self._smooth_face_tracking(FacialExpressionEngine.track_face(frame))

        # 4. Run the trained HDTF branch on its expected [B, 11, 3] landmark input.
        qlora_predictions = {}
        if self.model is not None:
            # The COCO task was trained on 256x256 RGB images with ImageNet
            # normalization. Reuse it at a lower cadence and let MediaPipe remain
            # the stable live tracker, using QLoRA to fill weak body joints.
            self._frame_index += 1
            if self._cached_coco is None or self._frame_index % 3 == 1:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
                    image_tensor = torch.from_numpy(((resized - mean) / std).transpose(2, 0, 1)).unsqueeze(0).to(self.device)
                    with torch.inference_mode():
                        pred = self.model(image_tensor, task="coco")[0].float().cpu().numpy()
                    if pred.shape == (17, 3) and np.isfinite(pred).all():
                        self._cached_coco = pred
                except Exception as exc:
                    if not self._model_inference_warning_shown:
                        print(f"[Pipeline Warning] COCO checkpoint inference failed: {exc}")
                        self._model_inference_warning_shown = True
            if self._cached_coco is not None:
                qlora_predictions["coco_keypoints"] = self._cached_coco.copy()
                smoothed_lm = self._blend_coco_pose(smoothed_lm, self._cached_coco, frame.shape[1], frame.shape[0])

        if self.model is not None and face_tracking is not None:
            try:
                facial_input = torch.from_numpy(face_tracking["model_landmarks"]).unsqueeze(0).to(self.device)
                with torch.inference_mode():
                    pred = self.model(facial_input, task="hdtf")[0].float().cpu().numpy()
                if pred.shape[0] == 4 and np.isfinite(pred).all():
                    qlora_predictions["hdtf_facial"] = {
                        "mouth_open_ratio": float(np.clip(pred[0], 0.0, 1.0)),
                        "lip_stretch": float(np.clip(pred[1], 0.0, 1.0)),
                        "eye_blink_ratio": float(np.clip(pred[2], 0.0, 1.0)),
                        "head_roll_deg": float(np.clip(pred[3], -45.0, 45.0)),
                    }
            except Exception as exc:
                if not self._model_inference_warning_shown:
                    print(f"[Pipeline Warning] HDTF checkpoint inference failed: {exc}")
                    self._model_inference_warning_shown = True

        # Compute Pillar 1 (Facial Expressions) & Pillar 2 (MoCap 3D Rig)
        expressions = FacialExpressionEngine.compute_expressions(
            smoothed_lm, frame=frame, face_tracking=face_tracking
        )
        mocap_data = MoCap3DRig.compute_mocap_rig(smoothed_lm)

        # Blend model output with direct MediaPipe measurements for stable live motion.
        if qlora_predictions.get("hdtf_facial"):
            for k, v in qlora_predictions["hdtf_facial"].items():
                if np.isfinite(v):
                    expressions[k] = float(np.clip(expressions.get(k, 0.0) * 0.65 + v * 0.35,
                                                   -45.0 if k == "head_roll_deg" else 0.0,
                                                   45.0 if k == "head_roll_deg" else 1.0))
            expressions["is_talking"] = expressions["mouth_open_ratio"] > 0.12

        # Pre-Recording Readiness & Live Animation Gate
        is_ready, status_msg, face_conf, body_conf = self.pose_estimator.validate_precheck(smoothed_lm)

        # 5. Render Live 3D Digital Human Avatar (Driven by QLoRA Model & 21-Keypoint Hands)
        is_recording = (self.recorder.state == RecordingState.RECORDING)
        model_status = "QLoRA HDTF Active" if qlora_predictions.get("hdtf_facial") else "MediaPipe Face Tracking"
        status_banner = f"{status_msg} | {model_status}" if not is_recording else f"RECORDING ({self.recorder.frame_count} frames)"
        posture = body_tracking["state"]
        posture_status = ("BODY: show hips and knees" if posture == "unknown"
                          else f"BODY: {posture}" + ("" if body_tracking["tracked"] else " (held)"))
        status_banner += f" | {posture_status}"

        avatar_preview = self.animator.render_canvas(
            smoothed_lm,
            status_text=status_banner,
            recording=is_recording,
            is_ready=is_ready,
            qlora_preds=qlora_predictions,
            hands_dict=hands_dict,
            expressions=expressions,
            face_tracking=face_tracking,
            hand_world=self.pose_estimator.hand_world_landmarks,
            body_tracking=body_tracking,
            frame=frame
        )

        # Build WebCam Preview with Bounding Box & Status Overlay
        camera_preview = frame.copy()
        if bbox is not None:
            camera_preview = self.detector.draw_bbox(camera_preview, bbox, yolo_conf)

        # On-Screen Pre-Recording Readiness Badge
        badge_color = config.COLOR_TEXT_OK if is_ready else config.COLOR_TEXT_WARN
        badge_text = "READY TO RECORD (3D Avatar Active)" if is_ready else f"TRACKING: {status_msg}"
        cv2.rectangle(camera_preview, (10, 10), (620, 60), (20, 24, 30), -1)
        cv2.putText(camera_preview, badge_text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.60, badge_color, 2)

        if is_recording:
            side_by_side = np.hstack([camera_preview, avatar_preview])
            export_frame = cv2.resize(side_by_side, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            self.recorder.record_frame(export_frame, smoothed_lm)

        telemetry = {
            "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
            "body_posture": {key: body_tracking[key]
                             for key in ("state", "tracked", "confidence", "coverage")},
            "yolo_confidence": yolo_conf,
            "pose_confidence": pose_conf,
            "face_confidence": face_conf,
            "body_confidence": body_conf,
            "is_ready": is_ready,
            "status_message": status_msg,
            "facial_expressions": expressions,
            "mocap_3d": mocap_data.get("joint_angles", {}),
            "state": self.recorder.state.value,
            "landmarks": smoothed_lm.tolist() if smoothed_lm is not None else None,
            "hands": {side: (points.tolist() if points is not None else None)
                      for side, points in (hands_dict or {}).items()},
            "face_tracking": ({
                "eye_gaze": face_tracking.get("eye_gaze"),
                "facial_pose_matrix": face_tracking.get("facial_pose_matrix"),
                "iris_centers": face_tracking.get("iris_centers"),
                "blendshapes": face_tracking.get("blendshapes", {}),
            } if face_tracking else None),
            "qlora_coco_active": "coco_keypoints" in qlora_predictions,
            "qlora_coco_keypoints": (qlora_predictions["coco_keypoints"].tolist()
                                     if "coco_keypoints" in qlora_predictions else None),
            "qlora_hdtf_output": qlora_predictions.get("hdtf_facial"),
            "qlora_pose_active": self.model is not None,
            "qlora_face_active": bool(qlora_predictions.get("hdtf_facial")),
            "face_tracking_active": face_tracking is not None
        }

        return camera_preview, avatar_preview, telemetry

    def close(self):
        try:
            self.pose_estimator.close()
        finally:
            try:
                FacialExpressionEngine.close()
            finally:
                self.animator.close()

    @staticmethod
    def _blend_coco_pose(landmarks, prediction, width, height):
        """Use QLoRA COCO joints to recover occluded limbs without overriding good MP tracks."""
        if landmarks is None:
            return None
        result = landmarks.copy()
        coco_to_mediapipe = {5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
                             11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28}
        for coco_idx, mp_idx in coco_to_mediapipe.items():
            x = float(prediction[coco_idx, 0]) * width / 256.0
            y = float(prediction[coco_idx, 1]) * height / 256.0
            visibility = float(np.clip(prediction[coco_idx, 2] / 2.0, 0.0, 1.0))
            if visibility < 0.35 or not np.isfinite([x, y]).all():
                continue
            x = float(np.clip(x, -width * 0.15, width * 1.15))
            y = float(np.clip(y, -height * 0.15, height * 1.15))
            current_visibility = float(result[mp_idx, 3])
            # Keep MediaPipe dominant when confident; let model help when obscured.
            model_weight = 0.12 if current_visibility >= 0.55 else 0.45 * visibility
            result[mp_idx, 0] = result[mp_idx, 0] * (1.0 - model_weight) + x * model_weight
            result[mp_idx, 1] = result[mp_idx, 1] * (1.0 - model_weight) + y * model_weight
            result[mp_idx, 3] = max(current_visibility, 0.55 * visibility)
        return result

    def _smooth_face_tracking(self, current):
        """Light temporal smoothing for face anchors while retaining quick expression response."""
        if current is None:
            self._face_track_ema = None
            return None
        if self._face_track_ema is None:
            self._face_track_ema = current
            return current

        alpha = 0.58
        previous = self._face_track_ema
        smoothed = dict(current)
        for key in ("center", "left_eye", "right_eye", "mouth_center", "eye_centers", "iris_centers"):
            if key in current and key in previous:
                a, b = np.asarray(current[key], dtype=np.float32), np.asarray(previous[key], dtype=np.float32)
                smoothed[key] = tuple(np.rint(alpha * a + (1.0 - alpha) * b).astype(int).tolist()) if a.ndim == 1 else tuple(
                    tuple(np.rint(alpha * x + (1.0 - alpha) * y).astype(int).tolist()) for x, y in zip(a, b)
                )
        for key in ("face_width", "face_height", "left_eye_width", "right_eye_width", "mouth_width", "mouth_height"):
            if key in current and key in previous:
                smoothed[key] = alpha * float(current[key]) + (1.0 - alpha) * float(previous[key])
        smoothed["model_landmarks"] = alpha * current["model_landmarks"] + (1.0 - alpha) * previous["model_landmarks"]
        current_blends = current.get("blendshapes", {})
        previous_blends = previous.get("blendshapes", {})
        smoothed["blendshapes"] = {
            key: alpha * float(value) + (1.0 - alpha) * float(previous_blends.get(key, value))
            for key, value in current_blends.items()
        }
        smoothed["expressions"] = {
            key: alpha * float(current["expressions"].get(key, 0.0))
                 + (1.0 - alpha) * float(previous["expressions"].get(key, 0.0))
            for key in ("mouth_open_ratio", "lip_stretch", "eye_blink_ratio", "head_roll_deg")
        }
        smoothed["expressions"]["is_talking"] = smoothed["expressions"]["mouth_open_ratio"] > 0.12
        self._face_track_ema = smoothed
        return smoothed
