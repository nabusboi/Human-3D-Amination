import os
import torch
import cv2
import numpy as np
from typing import Tuple, Dict, Optional
import config
from src.detector.yolo_detector import YOLOHumanDetector
from src.pose.mediapipe_pose import MediaPipePoseEstimator
from src.pose.facial_expression_engine import FacialExpressionEngine
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
        self.smoother = ExponentialMovingAverageSmoother(alpha=0.60)
        self.animator = Avatar3DAnimator(canvas_width=config.FRAME_WIDTH, canvas_height=config.FRAME_HEIGHT)
        self.recorder = MotionRecorder()

        # Load Trained Unified Multi-Task QLoRA Transformer Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
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

        if config.MIRROR_WEBCAM:
            frame = cv2.flip(frame, 1)

        # 1. YOLO Human Subject Bounding Box
        bbox, yolo_conf, _ = self.detector.detect_person(frame)

        # 2. MediaPipe 33 Body Keypoints + 21 Hand Keypoints
        landmarks_33, lm_dict, pose_conf, hands_dict = self.pose_estimator.process_frame(frame)

        # 3. Motion Smoothing
        smoothed_lm = self.smoother.smooth(landmarks_33) if landmarks_33 is not None else None

        # 4. Refine Keypoints with Unified QLoRA Model Inference (if active)
        qlora_predictions = {}
        if self.model is not None and frame is not None:
            try:
                # Prepare image tensor [1, 3, 256, 256]
                img_rgb = cv2.cvtColor(cv2.resize(frame, (256, 256)), cv2.COLOR_BGR2RGB)
                img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                img_tensor = img_tensor.to(self.device)

                with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    pred_coco = self.model(img_tensor, task="coco").cpu().numpy()[0]   # [17, 3]
                    pred_hdtf = self.model(img_tensor, task="hdtf").cpu().numpy()[0]   # [4]
                    pred_mocap = self.model(img_tensor, task="mocap").cpu().numpy()[0] # [31, 3]

                    qlora_predictions = {
                        "coco_keypoints": pred_coco,
                        "hdtf_facial": {
                            "mouth_open_ratio": float(pred_hdtf[0]),
                            "lip_stretch": float(pred_hdtf[1]),
                            "eye_blink_ratio": float(pred_hdtf[2]),
                            "head_roll_deg": float(pred_hdtf[3])
                        },
                        "mocap_3d": pred_mocap
                    }
            except Exception as e:
                pass

        # Compute Pillar 1 (Facial Expressions) & Pillar 2 (MoCap 3D Rig)
        expressions = FacialExpressionEngine.compute_expressions(smoothed_lm, frame=frame)
        mocap_data = MoCap3DRig.compute_mocap_rig(smoothed_lm)

        # Merge QLoRA predictions if active
        if qlora_predictions.get("hdtf_facial"):
            for k, v in qlora_predictions["hdtf_facial"].items():
                if not np.isnan(v):
                    expressions[k] = (expressions.get(k, 0.0) * 0.4 + v * 0.6)

        # Pre-Recording Readiness & Live Animation Gate
        is_ready, status_msg, face_conf, body_conf = self.pose_estimator.validate_precheck(smoothed_lm)

        # 5. Render Live 3D Digital Human Avatar (Driven by QLoRA Model & 21-Keypoint Hands)
        is_recording = (self.recorder.state == RecordingState.RECORDING)
        model_status = "QLoRA Model Active" if self.model is not None else "MediaPipe Active"
        status_banner = f"{status_msg} | {model_status}" if not is_recording else f"RECORDING ({self.recorder.frame_count} frames)"

        stickman_preview = self.animator.render_canvas(
            smoothed_lm,
            status_text=status_banner,
            recording=is_recording,
            is_ready=is_ready,
            qlora_preds=qlora_predictions,
            hands_dict=hands_dict,
            expressions=expressions,
            frame=frame
        )

        # Build WebCam Preview with Bounding Box & Status Overlay
        camera_preview = frame.copy()
        if bbox is not None:
            camera_preview = self.detector.draw_bbox(camera_preview, bbox, yolo_conf)

        # On-Screen Pre-Recording Readiness Badge
        badge_color = config.COLOR_TEXT_OK if is_ready else config.COLOR_TEXT_WARN
        badge_text = "READY TO RECORD (3D Avatar Active)" if is_ready else f"ANIMATION LOCKED: {status_msg}"
        cv2.rectangle(camera_preview, (10, 10), (620, 60), (20, 24, 30), -1)
        cv2.putText(camera_preview, badge_text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.60, badge_color, 2)

        # Record Frame if Recording Active
        if is_recording:
            side_by_side = np.hstack([camera_preview, stickman_preview])
            export_frame = cv2.resize(side_by_side, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            self.recorder.record_frame(export_frame, smoothed_lm)

        telemetry = {
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
            "qlora_pose_active": self.model is not None
        }

        return camera_preview, stickman_preview, telemetry
