"""Pipeline adapter for the native, skinned GLB avatar renderer."""

import cv2
import config
from src.animator.native_avatar_renderer import NativeAvatarRenderer


class Avatar3DAnimator:
    def __init__(self, canvas_width=config.FRAME_WIDTH, canvas_height=config.FRAME_HEIGHT):
        self.width, self.height = canvas_width, canvas_height
        self.renderer = NativeAvatarRenderer(canvas_width, canvas_height)

    def render_canvas(self, landmarks_33, status_text="", recording=False,
                      is_ready=False, qlora_preds=None, hands_dict=None,
                      expressions=None, face_tracking=None, frame=None, hand_world=None,
                      body_tracking=None):
        canvas = self.renderer.render(landmarks_33, hands_dict, expressions, face_tracking,
                                      hand_world=hand_world, body_tracking=body_tracking)
        # OpenCV adds status text only. Panda3D renders the character geometry.
        if status_text:
            cv2.rectangle(canvas, (0, self.height - 42), (self.width, self.height), (20, 18, 27), -1)
            cv2.putText(canvas, status_text, (15, self.height - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (150, 230, 170) if is_ready else (90, 185, 245), 1, cv2.LINE_AA)
        if recording:
            cv2.putText(canvas, "REC", (self.width - 75, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 70, 255), 2, cv2.LINE_AA)
        return canvas

    def toggle_view(self):
        self.renderer.toggle_view()

    def calibrate(self):
        self.renderer.calibrate()

    def close(self):
        self.renderer.close()
