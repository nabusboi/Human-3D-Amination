import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFilter
import config
from src.pose.facial_expression_engine import FacialExpressionEngine
from src.utils.mocap_rig import MoCap3DRig


class Avatar3DAnimator:
    """
    Polished 2.5D Cartoon / Pixar-style Avatar Animator.
    Renders a high-fidelity stylized 3D/2.5D character with smooth linear gradients,
    volumetric lighting, specularity, ambient drop shadows, anti-aliased geometry,
    and expressive facial & hand tracking.
    
    Compatible with the existing pipeline (YOLO, MediaPipe, QLoRA/Transformer, MotionRecorder).
    """

    def __init__(self, canvas_width: int = config.FRAME_WIDTH, canvas_height: int = config.FRAME_HEIGHT):
        self.width = canvas_width
        self.height = canvas_height

        # Pre-generate smooth studio gradient backdrop to avoid recalculating every frame
        self._backdrop_cache = self._create_studio_gradient_backdrop()

    def _create_studio_gradient_backdrop(self) -> np.ndarray:
        """
        Creates a rich, studio-lit dark purple vignette gradient backdrop (1280x720).
        """
        h, w = self.height, self.width
        # Vertical gradient from slightly darker purple top to deep studio violet bottom
        y_coords = np.linspace(0, 1, h, dtype=np.float32).reshape(-1, 1)
        top_color = np.array([125, 75, 95], dtype=np.float32)     # Deep mauve/purple BGR
        bottom_color = np.array([80, 45, 60], dtype=np.float32)   # Dark studio plum BGR
        grad = top_color * (1.0 - y_coords) + bottom_color * y_coords
        backdrop = np.repeat(grad[:, np.newaxis, :], w, axis=1).astype(np.uint8)

        # Center soft radial illumination spotlight
        cx, cy = w // 2, int(h * 0.45)
        y, x = np.ogrid[:h, :w]
        dist_sq = ((x - cx) / (w * 0.55)) ** 2 + ((y - cy) / (h * 0.55)) ** 2
        radial_glow = np.clip(1.0 - dist_sq * 0.65, 0.0, 1.0)
        spotlight = np.array([45, 25, 35], dtype=np.float32)

        backdrop_f = backdrop.astype(np.float32)
        backdrop_f += spotlight * radial_glow[:, :, np.newaxis]
        return np.clip(backdrop_f, 0, 255).astype(np.uint8)

    def render_canvas(self,
                      landmarks_33: Optional[np.ndarray],
                      status_text: str = "",
                      recording: bool = False,
                      is_ready: bool = False,
                      qlora_preds: Optional[Dict] = None,
                      hands_dict: Optional[Dict] = None,
                      expressions: Optional[Dict] = None,
                      frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Renders the 2.5D Stylized Avatar character on the studio backdrop.
        """
        canvas = self._backdrop_cache.copy()

        # Floor drop shadow beneath character
        self._draw_ambient_floor_shadow(canvas)

        if landmarks_33 is None:
            cv2.putText(canvas, "INITIALIZING 3D PIXAR AVATAR...",
                        (int(self.width * 0.22), int(self.height * 0.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 225, 235), 2, cv2.LINE_AA)
            return canvas

        # Extract expressions & mocap metrics
        if expressions is None:
            expressions = FacialExpressionEngine.compute_expressions(landmarks_33, frame=frame)
        mocap_data = MoCap3DRig.compute_mocap_rig(landmarks_33)

        # Merge QLoRA facial predictions if active
        if qlora_preds and "hdtf_facial" in qlora_preds:
            for k, v in qlora_preds["hdtf_facial"].items():
                if not np.isnan(v):
                    expressions[k] = v

        # Render avatar with smooth shading and layered depth
        self._render_pixar_character(canvas, landmarks_33, mocap_data, expressions, hands_dict)

        # Recording badge overlay
        if recording:
            cv2.circle(canvas, (self.width - 55, 35), 11, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (self.width - 55, 35), 15, (50, 50, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "REC", (self.width - 120, 43),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2, cv2.LINE_AA)

        # Status badge overlay
        badge_color = (110, 240, 150) if is_ready else (70, 180, 255)
        if status_text:
            # Elegant semi-transparent pill backdrop for text
            t_box_w = min(self.width - 40, int(len(status_text) * 11.5) + 30)
            overlay = canvas.copy()
            cv2.rectangle(overlay, (20, self.height - 48), (20 + t_box_w, self.height - 12), (25, 18, 28), -1)
            cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)
            cv2.putText(canvas, status_text, (32, self.height - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, badge_color, 2, cv2.LINE_AA)

        return canvas

    def _draw_ambient_floor_shadow(self, canvas: np.ndarray):
        """Draws a soft diffused contact shadow ellipse on the studio floor."""
        h, w = self.height, self.width
        overlay = canvas.copy()
        center = (w // 2, int(h * 0.90))
        axes = (int(w * 0.36), int(h * 0.075))
        # Deep contact shadow core
        cv2.ellipse(overlay, center, axes, 0, 0, 360, (40, 20, 30), -1, cv2.LINE_AA)
        # Soft blurred falloff
        cv2.ellipse(overlay, center, (int(axes[0] * 1.15), int(axes[1] * 1.3)), 0, 0, 360, (55, 30, 42), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

    def _render_pixar_character(self, canvas: np.ndarray, lm: np.ndarray,
                                mocap: Dict, expr: Dict, hands_dict: Optional[Dict]):
        """
        Renders the full 2.5D Pixar character with depth ordering,
        shading, volumetric limbs, hands, and expressive face.
        """
        l_sh, r_sh = lm[11], lm[12]
        l_hip, r_hip = lm[23], lm[24]
        l_el, r_el = lm[13], lm[14]
        l_wr, r_wr = lm[15], lm[16]
        l_kn, r_kn = lm[25], lm[26]
        l_an, r_an = lm[27], lm[28]
        nose = lm[0]

        sh_w = np.linalg.norm(l_sh[:2] - r_sh[:2])
        if sh_w < 15.0:
            sh_w = 120.0

        # Colors (BGR)
        SKIN_BASE = (175, 215, 255)
        PANTS_COLOR = (90, 85, 95)       # Charcoal/slate denim
        SHIRT_COLOR = (235, 130, 30)     # Vibrant Pixar royal blue

        # -------------------------------------------------------------
        # 1. LEGS & SNEAKERS (Rendered behind torso)
        # -------------------------------------------------------------
        # Left Leg - streamlined thickness
        if l_hip[3] > 0.2 and l_kn[3] > 0.2:
            self._render_shaded_capsule(canvas, (l_hip[0], l_hip[1]), (l_kn[0], l_kn[1]),
                                        r1=int(sh_w * 0.12), r2=int(sh_w * 0.10), color=PANTS_COLOR)
        if l_kn[3] > 0.2 and l_an[3] > 0.2:
            self._render_shaded_capsule(canvas, (l_kn[0], l_kn[1]), (l_an[0], l_an[1]),
                                        r1=int(sh_w * 0.10), r2=int(sh_w * 0.085), color=PANTS_COLOR)
            self._render_sneaker(canvas, (int(l_an[0]), int(l_an[1])), is_left=True, scale=sh_w * 0.15)

        # Right Leg - streamlined thickness
        if r_hip[3] > 0.2 and r_kn[3] > 0.2:
            self._render_shaded_capsule(canvas, (r_hip[0], r_hip[1]), (r_kn[0], r_kn[1]),
                                        r1=int(sh_w * 0.12), r2=int(sh_w * 0.10), color=PANTS_COLOR)
        if r_kn[3] > 0.2 and r_an[3] > 0.2:
            self._render_shaded_capsule(canvas, (r_kn[0], r_kn[1]), (r_an[0], r_an[1]),
                                        r1=int(sh_w * 0.10), r2=int(sh_w * 0.085), color=PANTS_COLOR)
            self._render_sneaker(canvas, (int(r_an[0]), int(r_an[1])), is_left=False, scale=sh_w * 0.15)

        # -------------------------------------------------------------
        # 2. POLO SHIRT TORSO (Volumetric shading, collar & buttons)
        # -------------------------------------------------------------
        chest_l = (int(l_sh[0] - sh_w * 0.08), int(l_sh[1]))
        chest_r = (int(r_sh[0] + sh_w * 0.08), int(r_sh[1]))
        hip_l = (int(l_hip[0] - sh_w * 0.05), int(l_hip[1]))
        hip_r = (int(r_hip[0] + sh_w * 0.05), int(r_hip[1]))

        # Smooth curved torso polygon with outline
        torso_pts = np.array([chest_l, chest_r, hip_r, hip_l], dtype=np.int32)
        # Shaded main torso
        cv2.fillConvexPoly(canvas, torso_pts, SHIRT_COLOR, lineType=cv2.LINE_AA)
        # Torso outer outline for clean cartoon definition
        cv2.polylines(canvas, [torso_pts], True, (min(255, SHIRT_COLOR[0] - 40),
                                                  min(255, SHIRT_COLOR[1] - 30),
                                                  min(255, SHIRT_COLOR[2] - 10)), 2, cv2.LINE_AA)

        # Volumetric chest highlight
        mid_chest = ((chest_l[0] + chest_r[0]) // 2, (chest_l[1] + hip_l[1]) // 2)
        torso_w = max(16, abs(chest_r[0] - chest_l[0]) // 2)
        torso_h = max(16, abs(hip_l[1] - chest_l[1]) // 2)
        cv2.ellipse(canvas, mid_chest, (int(torso_w * 0.50), int(torso_h * 0.65)),
                    0, 0, 360, (255, 160, 60), -1, lineType=cv2.LINE_AA)

        # Polo Collar V-neck line and buttons
        mid_sh = ((l_sh[0] + r_sh[0]) * 0.5, (l_sh[1] + r_sh[1]) * 0.5)
        collar_tip = (int(mid_sh[0]), int(mid_sh[1] + sh_w * 0.22))
        cv2.line(canvas, chest_l, collar_tip, (255, 205, 110), 2, cv2.LINE_AA)
        cv2.line(canvas, chest_r, collar_tip, (255, 205, 110), 2, cv2.LINE_AA)
        # Pearl buttons
        for b_i in range(2):
            b_y = int(collar_tip[1] + 10 + b_i * 12)
            cv2.circle(canvas, (int(mid_sh[0]), b_y), 3, (250, 250, 250), -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(mid_sh[0]), b_y), 3, (150, 150, 150), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 3. ARMS & 21-KEYPOINT HANDS (Left and Right) - Reduced thickness
        # -------------------------------------------------------------
        left_h_21 = hands_dict.get("left_hand") if hands_dict else None
        right_h_21 = hands_dict.get("right_hand") if hands_dict else None

        # Left Arm (Upper: polo sleeve, Lower: skin forearm)
        if l_sh[3] > 0.2 and l_el[3] > 0.2:
            self._render_shaded_capsule(canvas, (l_sh[0], l_sh[1]), (l_el[0], l_el[1]),
                                        r1=int(sh_w * 0.095), r2=int(sh_w * 0.075), color=SHIRT_COLOR)
        if l_el[3] > 0.2 and l_wr[3] > 0.2:
            self._render_shaded_capsule(canvas, (l_el[0], l_el[1]), (l_wr[0], l_wr[1]),
                                        r1=int(sh_w * 0.075), r2=int(sh_w * 0.055), color=SKIN_BASE)
            self._render_21keypoint_hand(canvas, wrist=l_wr, elbow=l_el, hand_21=left_h_21,
                                         pinky=lm[17], index=lm[19], thumb=lm[21],
                                         is_left=True, scale=sh_w * 0.12)

        # Right Arm (Upper: polo sleeve, Lower: skin forearm)
        if r_sh[3] > 0.2 and r_el[3] > 0.2:
            self._render_shaded_capsule(canvas, (r_sh[0], r_sh[1]), (r_el[0], r_el[1]),
                                        r1=int(sh_w * 0.095), r2=int(sh_w * 0.075), color=SHIRT_COLOR)
        if r_el[3] > 0.2 and r_wr[3] > 0.2:
            self._render_shaded_capsule(canvas, (r_el[0], r_el[1]), (r_wr[0], r_wr[1]),
                                        r1=int(sh_w * 0.075), r2=int(sh_w * 0.055), color=SKIN_BASE)
            self._render_21keypoint_hand(canvas, wrist=r_wr, elbow=r_el, hand_21=right_h_21,
                                         pinky=lm[18], index=lm[20], thumb=lm[22],
                                         is_left=False, scale=sh_w * 0.12)

        # -------------------------------------------------------------
        # 4. NECK COLUMN (Anchored smoothly under chin)
        # -------------------------------------------------------------
        head_r = max(34.0, sh_w * 0.35)
        neck_top = (int(nose[0]), int(nose[1] + head_r * 0.60))
        neck_bot = (int(mid_sh[0]), int(mid_sh[1]))
        neck_w = int(max(10, sh_w * 0.13))
        # Draw shaded neck
        cv2.line(canvas, neck_bot, neck_top, (155, 195, 235), neck_w, cv2.LINE_AA)
        cv2.line(canvas, neck_bot, neck_top, (185, 225, 255), max(2, neck_w // 2), cv2.LINE_AA)

        # -------------------------------------------------------------
        # 5. PIXAR HEAD & EXPRESSIVE FACE (Drawn on top)
        # -------------------------------------------------------------
        self._render_pixar_head(canvas, lm, expr, sh_w)

    # -------------------------------------------------------------------------
    # Geometric Shading & Helper Functions
    # -------------------------------------------------------------------------

    def _render_shaded_capsule(self, canvas: np.ndarray,
                               p1: Tuple[float, float], p2: Tuple[float, float],
                               r1: int, r2: int, color: Tuple[int, int, int]):
        """
        Renders a 3D-shaded capsule limb segment with highlight and shadow passes.
        """
        pt1 = np.array(p1, dtype=np.float32)
        pt2 = np.array(p2, dtype=np.float32)
        vec = pt2 - pt1
        dist = np.linalg.norm(vec)
        if dist < 1.0:
            return

        norm_vec = vec / dist
        perp_vec = np.array([-norm_vec[1], norm_vec[0]], dtype=np.float32)

        # Outer polygon
        v1 = pt1 + perp_vec * r1
        v2 = pt2 + perp_vec * r2
        v3 = pt2 - perp_vec * r2
        v4 = pt1 - perp_vec * r1
        pts = np.array([v1, v2, v3, v4], dtype=np.int32)

        # Base body fill
        cv2.fillConvexPoly(canvas, pts, color, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (int(pt1[0]), int(pt1[1])), r1, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (int(pt2[0]), int(pt2[1])), r2, color, -1, lineType=cv2.LINE_AA)

        # Specular light highlight (offset slightly towards light source)
        light_offset = perp_vec * 0.35
        h_pt1 = pt1 + light_offset * r1
        h_pt2 = pt2 + light_offset * r2
        h_color = (min(255, int(color[0] * 1.25) + 30),
                   min(255, int(color[1] * 1.25) + 30),
                   min(255, int(color[2] * 1.25) + 30))
        cv2.line(canvas, (int(h_pt1[0]), int(h_pt1[1])),
                 (int(h_pt2[0]), int(h_pt2[1])), h_color, max(2, int(r2 * 0.35)), cv2.LINE_AA)

        # Subtle dark ambient edge
        edge_color = (max(0, int(color[0] * 0.8)),
                      max(0, int(color[1] * 0.8)),
                      max(0, int(color[2] * 0.8)))
        cv2.polylines(canvas, [pts], True, edge_color, 1, lineType=cv2.LINE_AA)

    def _render_21keypoint_hand(self, canvas: np.ndarray,
                                wrist: np.ndarray, elbow: np.ndarray,
                                hand_21: Optional[np.ndarray],
                                pinky: np.ndarray, index: np.ndarray,
                                thumb: np.ndarray, is_left: bool, scale: float):
        """
        Renders an articulated 21-keypoint hand with palm base polygon and
        all 5 articulated 3D fingers sprouting from distinct knuckles.
        """
        skin_color = (175, 215, 255)
        skin_dark = (145, 185, 225)
        skin_highlight = (205, 235, 255)
        w_pt = wrist[:2]

        if hand_21 is not None and len(hand_21) == 21:
            raw_h_span = np.linalg.norm(hand_21[9][:2] - hand_21[0][:2])
            scale_factor = (scale * 0.78) / max(1.0, raw_h_span)

            pts_21 = {}
            for i_k in range(21):
                rel_v = (hand_21[i_k][:2] - hand_21[0][:2]) * scale_factor
                pts_21[i_k] = (int(w_pt[0] + rel_v[0]), int(w_pt[1] + rel_v[1]))

            # Palm base polygon (Wrist -> Thumb MCP -> Index MCP -> Middle MCP -> Ring MCP -> Pinky MCP)
            palm_poly = np.array(
                [pts_21[0], pts_21[1], pts_21[5], pts_21[9], pts_21[13], pts_21[17]],
                dtype=np.int32)
            cv2.fillConvexPoly(canvas, palm_poly, skin_color, lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [palm_poly], True, skin_dark, 2, lineType=cv2.LINE_AA)

            finger_chains = [
                [1, 2, 3, 4],     # Thumb
                [5, 6, 7, 8],     # Index
                [9, 10, 11, 12],  # Middle
                [13, 14, 15, 16], # Ring
                [17, 18, 19, 20], # Pinky
            ]
            finger_thick = max(4, int(scale * 0.16))
            for chain in finger_chains:
                prev_pt = pts_21[chain[0]]
                for j_idx in chain[1:]:
                    curr_pt = pts_21[j_idx]
                    cv2.line(canvas, prev_pt, curr_pt, skin_dark, finger_thick + 2, cv2.LINE_AA)
                    cv2.line(canvas, prev_pt, curr_pt, skin_color, finger_thick, cv2.LINE_AA)
                    cv2.circle(canvas, curr_pt, max(2, finger_thick // 2), skin_highlight, -1, cv2.LINE_AA)
                    prev_pt = curr_pt
            return

        # Directional fallback if MediaPipe Hand is partially out of frame
        e_pt = elbow[:2]
        dir_vec = w_pt - e_pt
        dist = np.linalg.norm(dir_vec)
        dir_vec = dir_vec / dist if dist > 0 else np.array([0, -1], dtype=np.float32)
        perp_vec = np.array([-dir_vec[1], dir_vec[0]], dtype=np.float32)

        palm_pt = w_pt + dir_vec * (scale * 0.18)
        p1 = (int(palm_pt[0]), int(palm_pt[1]))
        palm_r = max(3, int(scale * 0.10))
        thick = max(2, int(scale * 0.08))
        finger_len = scale * 0.40

        # Palm circular base with shading
        cv2.circle(canvas, p1, palm_r, skin_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, palm_r, skin_dark, 1, cv2.LINE_AA)

        # 5 fingers with natural anatomical fan spread
        # Thumb
        t_target = thumb[:2] if thumb[3] > 0.15 else (
            palm_pt + (perp_vec if is_left else -perp_vec) * finger_len * 0.70 + dir_vec * finger_len * 0.10)
        cv2.line(canvas, p1, (int(t_target[0]), int(t_target[1])), skin_color, thick + 1, cv2.LINE_AA)
        cv2.circle(canvas, (int(t_target[0]), int(t_target[1])), max(2, thick // 2), skin_highlight, -1, cv2.LINE_AA)

        # Index
        idx_target = index[:2] if index[3] > 0.15 else (
            palm_pt + dir_vec * finger_len * 0.95 + (perp_vec if is_left else -perp_vec) * finger_len * 0.30)
        cv2.line(canvas, p1, (int(idx_target[0]), int(idx_target[1])), skin_color, thick, cv2.LINE_AA)
        cv2.circle(canvas, (int(idx_target[0]), int(idx_target[1])), max(2, thick // 2), skin_highlight, -1, cv2.LINE_AA)

        # Middle
        mid_target = (
            palm_pt + dir_vec * finger_len * 1.05 + (perp_vec if is_left else -perp_vec) * finger_len * 0.05)
        cv2.line(canvas, p1, (int(mid_target[0]), int(mid_target[1])), skin_color, thick, cv2.LINE_AA)
        cv2.circle(canvas, (int(mid_target[0]), int(mid_target[1])), max(2, thick // 2), skin_highlight, -1, cv2.LINE_AA)

        # Ring
        ring_target = (
            palm_pt + dir_vec * finger_len * 0.95 - (perp_vec if is_left else -perp_vec) * finger_len * 0.20)
        cv2.line(canvas, p1, (int(ring_target[0]), int(ring_target[1])), skin_color, thick, cv2.LINE_AA)
        cv2.circle(canvas, (int(ring_target[0]), int(ring_target[1])), max(2, thick // 2), skin_highlight, -1, cv2.LINE_AA)

        # Pinky
        pnk_target = pinky[:2] if pinky[3] > 0.15 else (
            palm_pt + dir_vec * finger_len * 0.80 - (perp_vec if is_left else -perp_vec) * finger_len * 0.40)
        cv2.line(canvas, p1, (int(pnk_target[0]), int(pnk_target[1])), skin_color, thick, cv2.LINE_AA)
        cv2.circle(canvas, (int(pnk_target[0]), int(pnk_target[1])), max(2, thick // 2), skin_highlight, -1, cv2.LINE_AA)

    def _render_sneaker(self, canvas: np.ndarray, ankle: Tuple[int, int],
                        is_left: bool, scale: float):
        """Renders 3D stylized cyan & orange sneakers with white soles."""
        ax, ay = ankle
        offset_x = -int(scale * 0.85) if is_left else int(scale * 0.25)

        shoe_box = np.array([
            [ax + offset_x, ay],
            [ax + offset_x + int(scale * 1.25), ay],
            [ax + offset_x + int(scale * 1.35), ay + int(scale * 0.52)],
            [ax + offset_x - int(scale * 0.2), ay + int(scale * 0.52)],
        ], dtype=np.int32)

        # Cyan shoe body
        cv2.fillConvexPoly(canvas, shoe_box, (190, 215, 20), lineType=cv2.LINE_AA)
        # Orange heel badge
        cv2.ellipse(canvas,
                    (ax + offset_x + int(scale * 0.5), ay + int(scale * 0.25)),
                    (int(scale * 0.55), int(scale * 0.25)), 0, 0, 360, (20, 130, 255), -1, cv2.LINE_AA)
        # White thick sneaker sole
        cv2.line(canvas,
                 (ax + offset_x - int(scale * 0.2), ay + int(scale * 0.52)),
                 (ax + offset_x + int(scale * 1.35), ay + int(scale * 0.52)),
                 (250, 250, 250), max(3, int(scale * 0.12)), cv2.LINE_AA)

    def _render_pixar_head(self, canvas: np.ndarray, lm: np.ndarray,
                           expr: Dict, sh_w: float):
        """
        Renders the expressive 2.5D Pixar head with volumetric hair,
        beard, big anime/Disney eyes with pupil specular highlights, speaking jaw,
        and real-time head roll angle rotation.
        """
        nose = lm[0]
        if nose[3] < 0.2:
            return

        center = (float(nose[0]), float(nose[1]))
        head_r = int(max(34.0, sh_w * 0.35))

        # Head Roll Angle (radians and rotation helper)
        roll_deg = float(np.clip(expr.get("head_roll_deg", 0.0), -45.0, 45.0))
        roll_rad = np.radians(roll_deg)
        cos_r = np.cos(roll_rad)
        sin_r = np.sin(roll_rad)

        def rot_pt(dx: float, dy: float) -> Tuple[int, int]:
            """Rotate offset (dx, dy) around head center by roll angle."""
            rx = dx * cos_r - dy * sin_r
            ry = dx * sin_r + dy * cos_r
            return (int(center[0] + rx), int(center[1] + ry))

        # Color Palette
        skin_base = (175, 215, 255)
        skin_shadow = (145, 185, 225)
        hair_color = (35, 55, 80)        # Warm brown
        hair_highlight = (55, 80, 115)

        mouth_open = expr.get("mouth_open_ratio", 0.0)
        jaw_drop_offset = int(head_r * 0.25 * mouth_open)

        # 1. Ears with inner shading (Rotated with head)
        ear_r = int(head_r * 0.22)
        for sign in [-1, 1]:
            ear_c = rot_pt(sign * head_r * 0.94, 0)
            cv2.circle(canvas, ear_c, ear_r, skin_base, -1, cv2.LINE_AA)
            cv2.circle(canvas, ear_c, int(ear_r * 0.55), skin_shadow, -1, cv2.LINE_AA)

        # 2. Main Cranium / Head Sphere (Centered on nose)
        head_c = (int(center[0]), int(center[1]))
        cv2.circle(canvas, head_c, head_r, skin_base, -1, cv2.LINE_AA)

        # Soft transparent blush on cheekbones
        blush_overlay = canvas.copy()
        for sign in [-1, 1]:
            cheek_c = rot_pt(sign * head_r * 0.38, head_r * 0.12)
            cv2.circle(blush_overlay, cheek_c, int(head_r * 0.22), (170, 195, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(blush_overlay, 0.45, canvas, 0.55, 0, canvas)

        # 3. Dynamic Beard & Chin (Rotated with head)
        jaw_c = rot_pt(0, head_r * 0.46 + jaw_drop_offset)
        cv2.ellipse(canvas, jaw_c, (int(head_r * 0.88), int(head_r * 0.58)),
                    int(roll_deg), 0, 180, hair_color, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, rot_pt(0, head_r * 0.46 + jaw_drop_offset + 4),
                    (int(head_r * 0.78), int(head_r * 0.42)),
                    int(roll_deg), 0, 180, hair_highlight, -1, cv2.LINE_AA)

        # 4. Volumetric Hair Quiff (layered clumps rotated with head)
        hair_top = rot_pt(0, -head_r * 0.85)
        cv2.ellipse(canvas, hair_top, (int(head_r * 1.05), int(head_r * 0.58)),
                    int(roll_deg), 0, 360, hair_color, -1, cv2.LINE_AA)
        # Forehead swoop curl
        curl_c = rot_pt(-head_r * 0.15, -head_r * 0.95)
        cv2.ellipse(canvas, curl_c, (int(head_r * 0.70), int(head_r * 0.42)),
                    int(roll_deg - 15), 0, 360, hair_highlight, -1, cv2.LINE_AA)

        # 5. Mustache (drawn neatly above upper lip, rotated)
        mustache_y_off = head_r * 0.28 + jaw_drop_offset * 0.20
        mustache_c = rot_pt(0, mustache_y_off)
        cv2.ellipse(canvas, mustache_c,
                    (int(head_r * 0.44), int(head_r * 0.14)), int(roll_deg), 0, 180, hair_color, -1, cv2.LINE_AA)

        # 6. Expressive Pixar Eyes (Sclera + Iris + Specular Reflection + Eyelids)
        blink = expr.get("eye_blink_ratio", 0.0)
        eye_h = max(2, int(13 * (1.0 - blink)))
        eye_w = 9

        l_eye = rot_pt(-head_r * 0.36, -head_r * 0.16)
        r_eye = rot_pt(head_r * 0.36, -head_r * 0.16)

        if blink < 0.85:
            for e_pt in [l_eye, r_eye]:
                # White sclera
                cv2.ellipse(canvas, e_pt, (eye_w + 3, eye_h + 3), int(roll_deg), 0, 360, (250, 250, 250), -1, cv2.LINE_AA)
                # Dark animated iris
                cv2.ellipse(canvas, e_pt, (eye_w, eye_h), int(roll_deg), 0, 360, (30, 25, 25), -1, cv2.LINE_AA)
                # Pixar gleam specular highlight
                spec_pt = (int(e_pt[0] - 2 * cos_r), int(e_pt[1] - 2 * sin_r - max(1, eye_h // 3)))
                cv2.circle(canvas, spec_pt, 2, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            # Closed happy eyelid curve
            for e_pt in [l_eye, r_eye]:
                cv2.ellipse(canvas, e_pt, (eye_w + 2, 4), int(roll_deg), 0, 180, (50, 40, 40), 2, cv2.LINE_AA)

        # Eyebrows (Rotated with head)
        for sign in [-1, 1]:
            eb_c = rot_pt(sign * head_r * 0.36, -head_r * 0.32)
            eb_p1 = (int(eb_c[0] - sign * 10 * cos_r), int(eb_c[1] - sign * 10 * sin_r + 2))
            eb_p2 = (int(eb_c[0] + sign * 12 * cos_r), int(eb_c[1] + sign * 12 * sin_r - 3))
            cv2.line(canvas, eb_p1, eb_p2, hair_color, 3, cv2.LINE_AA)

        # 7. Animated Speaking Mouth (Fully closes when quiet, opens dynamically)
        mouth_y_off = head_r * 0.42 + jaw_drop_offset * 0.50
        mouth_center = rot_pt(0, mouth_y_off)
        base_w = head_r * 0.44
        mouth_w = int(base_w * (1.0 + expr.get("lip_stretch", 0.0) * 0.35))

        if mouth_open < 0.12:
            # Fully CLOSED mouth: clean friendly cartoon lip line
            lip_color = (130, 80, 100)
            p_left = rot_pt(-mouth_w, mouth_y_off)
            p_right = rot_pt(mouth_w, mouth_y_off)
            cv2.line(canvas, p_left, p_right, lip_color, 3, cv2.LINE_AA)
        else:
            # OPEN / TALKING mouth: scaled proportional to opening ratio
            open_norm = (mouth_open - 0.12) / 0.88
            mouth_h = int(3 + head_r * 0.55 * open_norm)

            # Deep crimson oral cavity (rotated with head)
            cv2.ellipse(canvas, mouth_center, (mouth_w, mouth_h),
                        int(roll_deg), 0, 360, (30, 20, 110), -1, cv2.LINE_AA)

            # White upper teeth row (visible when mouth opening is moderate/large)
            if mouth_h >= 8:
                teeth_h = min(7, int(mouth_h * 0.35))
                teeth_w = int(mouth_w * 0.72)
                t_box_poly = np.array([
                    rot_pt(-teeth_w, mouth_y_off - mouth_h + 2),
                    rot_pt(teeth_w, mouth_y_off - mouth_h + 2),
                    rot_pt(teeth_w, mouth_y_off - mouth_h + 2 + teeth_h),
                    rot_pt(-teeth_w, mouth_y_off - mouth_h + 2 + teeth_h)
                ], dtype=np.int32)
                cv2.fillConvexPoly(canvas, t_box_poly, (255, 255, 255), lineType=cv2.LINE_AA)

            # Smooth lip outline (rotated with head)
            cv2.ellipse(canvas, mouth_center, (mouth_w, mouth_h),
                        int(roll_deg), 0, 360, (180, 100, 125), 2, cv2.LINE_AA)
