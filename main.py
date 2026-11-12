import cv2
import numpy as np
import config
from src.pipeline import MovementDetectorPipeline
from src.recorder.motion_recorder import RecordingState
from src.utils.camera_capture import open_camera


def run_desktop_app():
    """Run webcam capture, tracking, avatar rendering, and recording in Python/OpenCV."""
    print("=" * 65)
    print("  HUMAN-TO-3D AVATAR MOVEMENT DETECTOR & TALKING HEAD ENGINE  ")
    print("=" * 65)
    print("HOTKEY CONTROLS:")
    print("  [SPACE] - Trigger face/body readiness check")
    print("  [R]     - Start / stop motion recording")
    print("  [V]     - Cycle automatic body / portrait / full body view")
    print("  [C]     - Recalibrate neutral head position")
    print("  [Q/ESC] - Quit")
    print("=" * 65)

    cap = open_camera(config.WEBCAM_INDEX)
    if cap is None:
        print("[Error] Python could not open the webcam. Check Windows camera access, close other camera apps, or set WEBCAM_INDEX in config.py.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)

    pipeline = None
    window_name = "Human-to-3D Avatar Movement Detector"

    try:
        pipeline = MovementDetectorPipeline()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.resizeWindow(window_name, 1440, 430)
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[Camera] Frame read failed; retrying.")
                if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                    break
                continue

            camera_preview, avatar_preview, telemetry = pipeline.process_frame(frame)
            h, w = camera_preview.shape[:2]
            avatar_resized = cv2.resize(avatar_preview, (w, h), interpolation=cv2.INTER_LINEAR)
            display = np.hstack([camera_preview, avatar_resized])
            banner = np.zeros((50, display.shape[1], 3), dtype=np.uint8)
            banner[:] = (15, 18, 22)
            state = telemetry.get("state", "IDLE")
            cv2.putText(banner, f"[SPACE] Precheck | [R] Record ({state}) | [V] View | [C] Calibrate | [Q] Quit",
                        (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 220, 240), 2, cv2.LINE_AA)
            composite = np.vstack([display, banner])
            # Win32 HighGUI can stretch WINDOW_KEEPRATIO when maximized.
            # Fit explicitly so both the person and avatar retain proportions.
            _, _, window_width, window_height = cv2.getWindowImageRect(window_name)
            if window_width > 0 and window_height > 0:
                scale = min(window_width / composite.shape[1],
                            window_height / composite.shape[0])
                fitted = cv2.resize(composite, (
                    max(1, round(composite.shape[1] * scale)),
                    max(1, round(composite.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
                presentation = np.zeros((window_height, window_width, 3), dtype=np.uint8)
                x = (window_width - fitted.shape[1]) // 2
                y = (window_height - fitted.shape[0]) // 2
                presentation[y:y + fitted.shape[0], x:x + fitted.shape[1]] = fitted
                composite = presentation
            cv2.imshow(window_name, composite)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("v"), ord("V")):
                pipeline.animator.toggle_view()
            if key in (ord("c"), ord("C")):
                pipeline.animator.calibrate()
            if key == 32:
                pipeline.recorder.trigger_precheck()
                print("[Action] Precheck requested.")
            if key in (ord("r"), ord("R")):
                if pipeline.recorder.state == RecordingState.RECORDING:
                    mp4_path, json_path = pipeline.recorder.stop_recording()
                    print(f"[Recording complete]\n  Video: {mp4_path}\n  Telemetry: {json_path}")
                elif telemetry.get("is_ready", False):
                    session = pipeline.recorder.start_recording()
                    print(f"[Recording started] {session}")
                else:
                    print(f"[Recording blocked] {telemetry.get('status_message', 'Waiting for face and body.')}")
    finally:
        cap.release()
        if pipeline is not None:
            if pipeline.recorder.state == RecordingState.RECORDING:
                pipeline.recorder.stop_recording()
            pipeline.close()
        cv2.destroyAllWindows()
        print("[Python] OpenCV webcam released.")


if __name__ == "__main__":
    run_desktop_app()
