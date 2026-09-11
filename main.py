import cv2
import sys
import numpy as np
import config
from src.pipeline import MovementDetectorPipeline
from src.recorder.motion_recorder import RecordingState


def run_desktop_app():
    """
    Interactive OpenCV Desktop GUI Application.
    Displays live webcam feed, 3D Digital Human Avatar preview, HDTF facial talking head, and pre-recording animation lock.
    """
    print("=" * 65)
    print("  HUMAN-TO-3D AVATAR MOVEMENT DETECTOR & TALKING HEAD ENGINE  ")
    print("=" * 65)
    print("HOTKEY CONTROLS:")
    print("  [SPACE] - Trigger Pre-recording Face & Body Validation Check")
    print("  [R]     - Start / Stop Motion Recording (.mp4 & .json export)")
    print("  [Q/ESC] - Quit Application")
    print("=" * 65)

    # Initialize Camera Capture
    cap = cv2.VideoCapture(config.WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)

    if not cap.isOpened():
        print(f"[Error] Failed to open camera device at index {config.WEBCAM_INDEX}.")
        sys.exit(1)

    # Instantiate Movement Pipeline
    pipeline = MovementDetectorPipeline()
    window_name = "Human-to-3D Avatar Movement Detector & Talking Head Engine"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Error] Failed to read frame from camera stream.")
                break

            # Process frame through 3D Avatar pipeline
            cam_preview, avatar_preview, telemetry = pipeline.process_frame(frame)

            # Combine camera feed and 3D avatar preview side-by-side
            h, w, _ = cam_preview.shape
            avatar_resized = cv2.resize(avatar_preview, (w, h))
            dual_display = np.hstack([cam_preview, avatar_resized])

            # Render Bottom Control Banner
            banner = np.zeros((50, dual_display.shape[1], 3), dtype=np.uint8)
            banner[:] = (15, 18, 22)
            
            state_str = telemetry.get("state", "IDLE")
            banner_text = f"[SPACE] Precheck  |  [R] Record ({state_str})  |  [Q] Quit"
            cv2.putText(banner, banner_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 220, 240), 2)
            
            full_window = np.vstack([dual_display, banner])
            cv2.imshow(window_name, full_window)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break

            elif key == 32:  # SPACE bar
                pipeline.recorder.trigger_precheck()
                print("[Action] Triggered pre-recording face & body validation check.")

            elif key == ord('r') or key == ord('R'):
                if pipeline.recorder.state == RecordingState.RECORDING:
                    mp4_path, json_path = pipeline.recorder.stop_recording()
                    print(f"\n[RECORDING COMPLETE]")
                    print(f"  Recorded MP4 Video: {mp4_path}")
                    print(f"  Keypoint Telemetry: {json_path}\n")
                else:
                    # Enforce Mandatory Pre-Recording Live Animation Gate
                    if telemetry.get("is_ready", False):
                        session_id = pipeline.recorder.start_recording()
                        print(f"\n[RECORDING STARTED] Session ID: {session_id}\n")
                    else:
                        print(f"\n[RECORDING BLOCKED] Cannot start recording!")
                        print(f"Reason: Live 3D Avatar animation is not fully active.")
                        print(f"Status: {telemetry.get('status_message')}")
                        print("Please stand in camera view until face & body landmarks show READY TO RECORD.\n")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[System] OpenCV camera stream and windows released.")


if __name__ == "__main__":
    run_desktop_app()
