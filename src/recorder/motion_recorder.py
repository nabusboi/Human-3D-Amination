import os
import time
import json
import cv2
import numpy as np
from enum import Enum
from typing import Optional, Dict, List, Tuple
import config


class RecordingState(Enum):
    IDLE = "IDLE"
    VALIDATION = "VALIDATION"
    RECORDING = "RECORDING"
    EXPORTED = "EXPORTED"


class MotionRecorder:
    """
    Motion Recorder State Machine.
    Manages pre-recording check, frame video writing (.mp4), and keyframe motion telemetry export (.json).
    """

    def __init__(self, output_dir: str = config.OUTPUT_DIR):
        self.output_dir = output_dir
        self.state = RecordingState.IDLE

        self.video_writer: Optional[cv2.VideoWriter] = None
        self.current_session_id: str = ""
        self.recorded_keyframes: List[Dict] = []
        self.start_time: float = 0.0
        self.frame_count: int = 0
        
        self.last_mp4_path: str = ""
        self.last_json_path: str = ""

    def trigger_precheck(self):
        """Switches state from IDLE to VALIDATION mode to check face/body readiness."""
        self.state = RecordingState.VALIDATION

    def start_recording(self, fps: int = config.FPS, width: int = config.FRAME_WIDTH, height: int = config.FRAME_HEIGHT) -> str:
        """Starts recording session, opens MP4 VideoWriter, initializes keyframe telemetry."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.current_session_id = f"motion_session_{timestamp}"
        
        mp4_filename = f"{self.current_session_id}.mp4"
        self.last_mp4_path = os.path.join(self.output_dir, mp4_filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.last_mp4_path, fourcc, fps, (width, height))

        self.recorded_keyframes = []
        self.start_time = time.time()
        self.frame_count = 0
        self.state = RecordingState.RECORDING

        print(f"[Recorder] Started motion recording session: {self.current_session_id}")
        return self.current_session_id

    def record_frame(self, rendered_frame: np.ndarray, landmarks_33: Optional[np.ndarray]):
        """Records a single video frame and appends keypoint landmark telemetry."""
        if self.state != RecordingState.RECORDING:
            return

        if self.video_writer is not None and rendered_frame is not None:
            self.video_writer.write(rendered_frame)

        elapsed = time.time() - self.start_time
        keyframe_data = {
            "frame_idx": self.frame_count,
            "timestamp_sec": round(elapsed, 4),
            "keypoints": landmarks_33.tolist() if landmarks_33 is not None else []
        }
        self.recorded_keyframes.append(keyframe_data)
        self.frame_count += 1

    def stop_recording() -> Tuple[str, str]:
        """Stops recording session, releases MP4 writer, and exports telemetry to JSON file."""
        if self.state != RecordingState.RECORDING:
            return "", ""

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        json_filename = f"{self.current_session_id}_telemetry.json"
        self.last_json_path = os.path.join(self.output_dir, json_filename)

        metadata = {
            "session_id": self.current_session_id,
            "total_frames": self.frame_count,
            "duration_sec": round(time.time() - self.start_time, 2),
            "fps": config.FPS,
            "keypoints_topology": "MediaPipe_33_Body_Keypoints",
            "keyframes": self.recorded_keyframes
        }

        with open(self.last_json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.state = RecordingState.EXPORTED
        print(f"[Recorder] Exported MP4 video to: {self.last_mp4_path}")
        print(f"[Recorder] Exported JSON telemetry to: {self.last_json_path}")
        return self.last_mp4_path, self.last_json_path

    def cancel(self):
        """Cancels recording or validation mode."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.state = RecordingState.IDLE
