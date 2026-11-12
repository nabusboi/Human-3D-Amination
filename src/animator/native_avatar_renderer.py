"""Native Panda3D rendering returned as BGR frames for the Python desktop UI."""

import math
import time

import cv2
import numpy as np
from direct.actor.Actor import Actor
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight, CharacterJoint, DirectionalLight, GraphicsOutput,
    NodePath, Quat, Texture, Vec3, load_prc_file_data,
)

import config
from src.animator.gltf_asset import load_avatar


class NativeAvatarRenderer:
    BODY_BONES = (
        ("LeftArm", "LeftForeArm", 11, 13),
        ("LeftForeArm", "LeftHand", 13, 15),
        ("RightArm", "RightForeArm", 12, 14),
        ("RightForeArm", "RightHand", 14, 16),
        ("LeftUpLeg", "LeftLeg", 23, 25),
        ("LeftLeg", "LeftFoot", 25, 27),
        ("LeftFoot", "LeftToeBase", 27, 31),
        ("RightUpLeg", "RightLeg", 24, 26),
        ("RightLeg", "RightFoot", 26, 28),
        ("RightFoot", "RightToeBase", 28, 32),
    )
    LEG_BONES = {
        "LeftUpLeg": "LeftLeg", "LeftLeg": "LeftFoot", "LeftFoot": "LeftToeBase",
        "RightUpLeg": "RightLeg", "RightLeg": "RightFoot", "RightFoot": "RightToeBase",
    }

    def __init__(self, width, height):
        self.width, self.height = width, height
        self.started = self.last_time = time.monotonic()
        self.base = None
        load_prc_file_data("native-avatar", (
            f"window-type offscreen\nwin-size {width} {height}\n"
            "audio-library-name null\nsync-video false\n"
            "textures-power-2 none\nmodel-cache-dir\n"
            "framebuffer-multisample true\nmultisamples 4\n"
        ))
        try:
            self.base = ShowBase(windowType="offscreen")
            if self.base.win is None:
                raise RuntimeError("Cannot create the native OpenGL avatar renderer.")
            self.base.disableMouse()
            self.base.setBackgroundColor(0.12, 0.10, 0.18, 1)
            source, morph_names = load_avatar(config.AVATAR_MODEL_PATH,
                                              config.AVATAR_IDLE_PATH)
            self.actor = Actor(source)
            self.actor.reparentTo(self.base.render)
            # A detached actor samples the idle clip. Its joints provide the
            # fallback pose; only the visible actor is drawn.
            self.idle = Actor(source)
            self.animation = next(iter(self.idle.getAnimNames()), None)
            if self.animation:
                self.idle.pose(self.animation, 0)
                self.idle.update(force=True)
            self.character = self.actor.find("**/+Character").node()
            self.bundle = self.character.getBundle(0)
            self.rig = NodePath("avatar-controls")
            self.controls, self.rest, self.idle_joints = {}, {}, {}
            idle_character = self.idle.find("**/+Character").node()

            def build(part, parent):
                for child in part.getChildren():
                    next_parent = parent
                    if isinstance(child, CharacterJoint):
                        name = child.getName()
                        control = parent.attachNewNode(name)
                        control.setMat(child.getDefaultValue())
                        self.controls[name] = control
                        self.rest[name] = control.getTransform()
                        self.idle_joints[name] = idle_character.findJoint(name)
                        self.bundle.controlJoint(name, control.node())
                        next_parent = control
                    build(child, next_parent)

            build(self.bundle, self.rig)
            self.previous_quats = {}
            self.leg_directions = {
                name: self.controls[child].getPos(self.rig) - self.controls[name].getPos(self.rig)
                for name, child in self.LEG_BONES.items()
            }
            # Capture the idle (standing) animation's leg bone directions.
            # These are used when standing is detected but legs are off-camera,
            # so the avatar holds a natural standing pose instead of bind-pose
            # bone offsets which may point in unnatural directions.
            self.standing_leg_directions = dict(self.leg_directions)
            if self.animation:
                saved = {name: control.getMat() for name, control in self.controls.items()}
                for name, control in self.controls.items():
                    joint = self.idle_joints[name]
                    if joint is not None:
                        control.setMat(joint.getValue())
                for bone, child_bone in self.LEG_BONES.items():
                    direction = (self.controls[child_bone].getPos(self.rig)
                                 - self.controls[bone].getPos(self.rig))
                    if direction.lengthSquared() > 1e-8:
                        self.standing_leg_directions[bone] = direction
                for name, control in self.controls.items():
                    control.setMat(saved[name])
            self.body_torso = self.body_hips = None
            self.hips_origin = Vec3(self.controls["Hips"].getPos(self.rig))
            self.root_offset = Vec3(0, 0, 0)
            feet = [self.controls[side + "Foot"].getPos(self.rig) for side in ("Left", "Right")]
            self.floor_ankle_z = min(point.z for point in feet)
            self.feet_center_y = sum(point.y for point in feet) * 0.5
            self.hand_targets = {}
            self.hand_seen = {}
            self.hand_rotations = {}
            self.wrist_targets = {}
            self.elbow_poles = {}
            self.positioned_arms = set()
            self.sliders = {name: self.character.findSlider(name) for name in morph_names}
            self.slider_values = {}
            self.face_reference = None
            self.face_angles = np.zeros(3)
            self.view = config.AVATAR_VIEW
            self._light_scene()
            self.actor.update(force=True)
            low, high = self.actor.getTightBounds()
            self.model_center = (low + high) * 0.5
            self.model_height = max(0.1, high.z - low.z)
            self.model_top = high.z
            self.model_floor = low.z
            head = self.controls.get("Head")
            self.face_anchor = (Vec3(head.getPos(self.rig)) if head is not None
                                else Vec3(self.model_center))
            self.face_anchor.z += self.model_height * 0.035
            self.face_anchor.y -= self.model_height * 0.045
            self.head_top_offset = high.z - self.controls["Head"].getPos(self.rig).z
            self._frame_camera()
            self.texture = Texture("avatar-frame")
            self.texture.setKeepRamImage(True)
            self.base.win.addRenderTexture(self.texture, GraphicsOutput.RTMCopyRam,
                                          GraphicsOutput.RTPColor)
            self.base.graphicsEngine.renderFrame()
            print(f"[Avatar] Native GLB active: {config.AVATAR_MODEL_PATH}")
            print(f"[Avatar] {len(self.controls)} bones, {len(self.sliders)} facial targets.")
        except Exception:
            self.close()
            raise

    def _light_scene(self):
        ambient = self.base.render.attachNewNode(AmbientLight("ambient"))
        ambient.node().setColor((0.58, 0.58, 0.62, 1))
        self.base.render.setLight(ambient)
        for name, color, hpr in (
            ("key", (0.85, 0.80, 0.74, 1), (-25, -25, 0)),
            ("fill", (0.30, 0.36, 0.48, 1), (140, -15, 0)),
        ):
            light = self.base.render.attachNewNode(DirectionalLight(name))
            light.node().setColor(color)
            light.setHpr(*hpr)
            self.base.render.setLight(light)

    def _frame_camera(self):
        center = Vec3(self.model_center)
        height = self.model_height
        if self.view in ("auto", "upper"):
            visible_height = height * 0.34
            center.z = self.model_top - visible_height * 0.40
            center.y = self.face_anchor.y
        else:
            visible_height = height * 1.28
        self.camera_height = self.camera_target_height = visible_height
        self.camera_center = center
        self.camera_target_z = center.z
        self.camera_face_seen = time.monotonic()
        self._place_camera()

    def _place_camera(self):
        center = self.camera_center
        vertical_fov = 35.0
        horizontal_fov = math.degrees(2 * math.atan(
            math.tan(math.radians(vertical_fov / 2)) * self.width / self.height))
        distance = self.camera_height / (2 * math.tan(math.radians(vertical_fov / 2)))
        self.base.camLens.setFov(horizontal_fov, vertical_fov)
        self.base.camLens.setNearFar(0.01, max(20, distance * 10))
        self.base.camera.setPos(center.x, center.y - distance, center.z)
        self.base.camera.lookAt(center)

    def _current_face_anchor(self):
        anchor = Vec3(self.controls["Head"].getPos(self.rig))
        anchor.z += self.model_height * 0.035
        anchor.y -= self.model_height * 0.045
        return anchor

    def _update_camera(self, face, dt, body_tracking=None):
        if not config.AVATAR_AUTO_FRAME:
            return
        anchor = self._current_face_anchor()
        top = self.controls["Head"].getPos(self.rig).z + self.head_top_offset
        coverage = (body_tracking or {}).get("coverage", "portrait")
        if self.view == "full":
            coverage = "full"
        elif self.view == "upper":
            coverage = "portrait"
        if coverage != "portrait":
            if coverage == "waist":
                bottom = self.controls["Hips"].getPos(self.rig).z - self.model_height * 0.12
            elif coverage == "knees":
                bottom = min(self.controls[side + "Leg"].getPos(self.rig).z
                             for side in ("Left", "Right")) - self.model_height * 0.10
            else:
                bottom = self.model_floor - self.model_height * 0.04
            self.camera_target_height = max(self.model_height * 0.38, (top - bottom) / 0.80)
            self.camera_target_z = (top + bottom) * 0.5
            self.camera_face_seen = time.monotonic()
        elif face:
            face_height = float(face.get("face_height", 0))
            center = np.asarray(face.get("center", ()), dtype=float)
            if (np.isfinite(face_height) and face_height > self.height * 0.08
                    and center.shape == (2,) and np.isfinite(center).all()):
                # Match face size, rather than trying to fit unseen legs into
                # a seated webcam shot. Allow space above the avatar's hair.
                fraction = face_height / self.height
                max_height = 0.65 if self.view == "auto" else 0.38
                target = float(np.clip(self.model_height * 0.125 / fraction,
                                       self.model_height * 0.24,
                                       self.model_height * max_height))
                self.camera_face_seen = time.monotonic()
                if abs(target - self.camera_target_height) > self.camera_target_height * 0.035:
                    self.camera_target_height = target
                target_z = anchor.z + (center[1] / self.height - 0.5) * target
                # Preserve the person's vertical position; only reserve a
                # small margin above the hair instead of pinning it to the top.
                target_z = max(float(target_z), top - target * 0.46)
                if abs(target_z - self.camera_target_z) > self.model_height * 0.007:
                    self.camera_target_z = target_z
        if time.monotonic() - self.camera_face_seen > 0.8:
            self.camera_target_height = self.model_height * 0.34
            self.camera_target_z = top - self.camera_target_height * 0.40
        # Retain the last framing through brief occlusion. Slow camera motion
        # keeps facial expressions and small head tilts from pumping the zoom.
        alpha = 1 - math.exp(-dt / 0.40)
        self.camera_height += (self.camera_target_height - self.camera_height) * alpha
        self.camera_center.z += (self.camera_target_z - self.camera_center.z) * alpha
        self.camera_center.y += (anchor.y - self.camera_center.y) * alpha
        self._place_camera()

    def toggle_view(self):
        views = ("auto", "upper", "full")
        self.view = views[(views.index(self.view) + 1) % len(views)] if self.view in views else "auto"
        self._frame_camera()

    def calibrate(self):
        self.face_reference = None

    @staticmethod
    def _blend_quat(previous, target, alpha):
        values = np.array([target[i] for i in range(4)])
        old = np.array([previous[i] for i in range(4)])
        if np.dot(old, values) < 0:
            values = -values
        values = old * (1 - alpha) + values * alpha
        values /= max(float(np.linalg.norm(values)), 1e-8)
        return Quat(*map(float, values))

    @staticmethod
    def _swing(source, target):
        source, target = Vec3(source), Vec3(target)
        source.normalize()
        target.normalize()
        dot = max(-1.0, min(1.0, source.dot(target)))
        if dot < -0.9999:
            axis = source.cross(Vec3(1, 0, 0))
            if axis.lengthSquared() < 1e-8:
                axis = source.cross(Vec3(0, 1, 0))
            axis.normalize()
            q = Quat()
            q.setFromAxisAngle(180, axis)
            return q
        cross = source.cross(target)
        q = Quat(1 + dot, cross.x, cross.y, cross.z)
        q.normalize()
        return q

    def _aim(self, name, child_name, direction):
        if name not in self.controls or child_name not in self.controls:
            return
        vector = Vec3(*map(float, direction))
        if vector.lengthSquared() < 1e-8:
            return
        control, child = self.controls[name], self.controls[child_name]
        parent_rotation = control.getParent().getQuat(self.rig)
        target = parent_rotation.conjugate().xform(vector)
        rotation = control.getQuat()
        current = rotation.xform(child.getPos())
        if current.lengthSquared() > 1e-8:
            control.setQuat(rotation * self._swing(current, target))

    def _direction(self, a, b):
        # MediaPipe gives pixel x/y but normalized z. Put all axes in pixel units.
        delta = np.asarray(b[:3]) - np.asarray(a[:3])
        return (delta[0], delta[2] * self.width, -delta[1])

    def _update_posture(self, tracking):
        if tracking is None:
            return
        self.leg_directions.update(tracking.get("leg_directions", {}))
        torso = tracking.get("torso_direction")
        across = tracking.get("hip_direction")
        if torso is not None and torso[2] > 0.35:
            self.body_torso = Vec3(*map(float, torso))
        if across is not None:
            self.body_hips = Vec3(*map(float, across))
        hips = self.controls["Hips"]
        hips.setTransform(self.rest["Hips"])
        hips.setPos(self.rig, self.hips_origin + self.root_offset)
        if self.body_hips is not None:
            heading = math.degrees(math.atan2(self.body_hips.y, self.body_hips.x))
            delta = Quat()
            delta.setFromAxisAngle(heading, Vec3(0, 0, 1))
            hips.setQuat(hips.getQuat() * delta)
        if self.body_torso is not None:
            for bone, child in (("Spine", "Spine1"), ("Spine1", "Spine2"), ("Spine2", "Neck")):
                self._aim(bone, child, self.body_torso)
        # Retain the last observed world directions through occlusion. Resetting
        # to the standing idle clip every frame used to erase the seated pose.
        # When confirmed standing but a bone has no tracked direction, use the
        # idle animation's natural standing pose instead of bind-pose offsets.
        state = (tracking or {}).get("state", "unknown")
        tracked_dirs = (tracking or {}).get("leg_directions", {})
        for bone, child in self.LEG_BONES.items():
            if state == "standing" and bone not in tracked_dirs:
                self._aim(bone, child, self.standing_leg_directions[bone])
            else:
                self._aim(bone, child, self.leg_directions[bone])

    def _ground_body(self, dt):
        feet = [self.controls[side + "Foot"].getPos(self.rig) for side in ("Left", "Right")]
        target = Vec3(self.root_offset)
        target.z += self.floor_ankle_z - min(point.z for point in feet)
        target.y += self.feet_center_y - sum(point.y for point in feet) * 0.5
        target.z = float(np.clip(target.z, -self.model_height * 0.62, self.model_height * 0.12))
        target.y = float(np.clip(target.y, -self.model_height * 0.45, self.model_height * 0.45))
        self.root_offset += (target - self.root_offset) * (1 - math.exp(-dt / 0.12))
        self.controls["Hips"].setPos(self.rig, self.hips_origin + self.root_offset)

    def _update_body(self, landmarks, body_tracking=None):
        points = np.asarray(landmarks) if landmarks is not None else np.empty(0)
        if points.shape != (33, 4) or not np.isfinite(points).all():
            return
        visible = ((points[:, 3] >= 0.55) & (points[:, 0] >= 0)
                   & (points[:, 0] < self.width) & (points[:, 1] >= 0)
                   & (points[:, 1] < self.height))
        if self.body_torso is None and np.all(visible[[11, 12, 23, 24]]):
            hips = (points[23] + points[24]) * 0.5
            shoulders = (points[11] + points[12]) * 0.5
            self._aim("Spine2", "Neck", self._direction(hips, shoulders))
        arm_directions = (body_tracking or {}).get("arm_directions", {})
        for name, child, start, end in self.BODY_BONES:
            if body_tracking is not None and name in self.LEG_BONES:
                continue
            if name in arm_directions:
                self._aim(name, child, arm_directions[name])
            elif visible[start] and visible[end]:
                self._aim(name, child, self._direction(points[start], points[end]))

    def _orient_palm(self, side, positions):
        """Match both the palm's length and width, including wrist twist."""
        name = side + "Hand"
        middle, index, pinky = (side + "Hand" + digit + "1"
                                for digit in ("Middle", "Index", "Pinky"))
        if any(n not in self.controls for n in (name, middle, index, pinky)):
            return
        forward = Vec3(*map(float, positions[9] - positions[0]))
        across = Vec3(*map(float, positions[5] - positions[17]))
        if forward.lengthSquared() < 1e-10 or across.lengthSquared() < 1e-10:
            return
        self._aim(name, middle, forward)
        control = self.controls[name]
        parent = control.getParent()
        inverse = parent.getQuat(self.rig).conjugate()
        axis = inverse.xform(forward)
        axis.normalize()
        target = inverse.xform(across)
        current = self.controls[index].getPos(parent) - self.controls[pinky].getPos(parent)
        current -= axis * current.dot(axis)
        target -= axis * target.dot(axis)
        if min(current.lengthSquared(), target.lengthSquared()) < 1e-10:
            return
        current.normalize()
        target.normalize()
        angle = math.degrees(math.atan2(axis.dot(current.cross(target)), current.dot(target)))
        twist = Quat()
        twist.setFromAxisAngle(angle, axis)
        control.setQuat(control.getQuat() * twist)

    def _position_hands(self, landmarks, hands, contact=False):
        """Solve fixed-length arms from wrists even with cropped elbows.

        Hand depth is relative to its own wrist, so it cannot be used as body
        depth. Use shoulder-relative image coordinates and bounded forward
        reach instead. An observed elbow supplies the bend plane.
        """
        points = np.asarray(landmarks) if landmarks is not None else np.empty(0)
        if points.shape != (33, 4) or not np.isfinite(points).all():
            return
        if min(points[11, 3], points[12, 3]) < 0.5:
            return
        pixel_span = np.linalg.norm(points[11, :2] - points[12, :2])
        left = self.controls["LeftArm"].getPos(self.rig)
        right = self.controls["RightArm"].getPos(self.rig)
        if pixel_span < self.width * 0.08:
            return
        scale = (left - right).length() / pixel_span
        now = time.monotonic()
        for side, key, shoulder_index, elbow_index in (
                ("Left", "left_hand", 11, 13), ("Right", "right_hand", 12, 14)):
            hand = np.asarray((hands or {}).get(key, ()), dtype=float)
            if hand.shape != (21, 3) or not np.isfinite(hand).all():
                continue
            shoulder = self.controls[side + "Arm"].getPos(self.rig)
            upper = self.controls[side + "ForeArm"].getPos().length()
            lower = self.controls[side + "Hand"].getPos().length()
            reach = upper + lower
            delta = (hand[0, :2] - points[shoulder_index, :2]) * scale
            # A palm approaching the lens grows; allow modest forward reach,
            # without feeding its wrist-relative z into the arm solver.
            palm_pixels = np.linalg.norm(hand[9, :2] - hand[0, :2])
            forward = float(np.clip(palm_pixels * scale, 0.08, reach * 0.45))
            offset = Vec3(float(delta[0]), -forward, float(-delta[1]))
            distance = float(np.clip(offset.length(), reach * 0.32, reach * 0.97))
            offset.normalize()
            axis = offset
            target = shoulder + axis * distance
            previous = self.wrist_targets.get(key)
            if previous is not None and now - previous[1] < 0.4:
                if (target - previous[0]).length() < reach * 0.012:
                    target = Vec3(previous[0])
                else:
                    blend = 1 - math.exp(-max(0.001, now - previous[1]) / 0.09)
                    target = previous[0] + (target - previous[0]) * blend
            # Keep the entire palm in front of the torso, not merely the wrist.
            # Use the body's facing direction so this also follows torso turns.
            front = self.controls["Hips"].getQuat(self.rig).xform(Vec3(0, -1, 0))
            front.normalize()
            chest = self.controls["Spine2"].getPos(self.rig)
            hips = self.controls["Hips"].getPos(self.rig)
            near_torso = (abs(target.x - chest.x) < (left - right).length() * 0.75
                          and hips.z - 0.05 < target.z < chest.z + self.model_height * 0.16)
            if contact or near_torso:
                clearance = self.model_height * 0.14
                # Intersect the reachable sphere with the chest clearance
                # plane. Clamping reach afterwards could push a hand inside.
                depth = max((target - shoulder).dot(front),
                            clearance - (shoulder - chest).dot(front))
                depth = min(depth, reach * 0.90)
                lateral = target - shoulder - front * (target - shoulder).dot(front)
                radius = math.sqrt(max(0, (reach * 0.97) ** 2 - depth ** 2))
                if lateral.length() > radius:
                    lateral.normalize()
                    lateral *= radius
                target = shoulder + front * depth + lateral
            offset = target - shoulder
            distance = float(np.clip(offset.length(), reach * 0.32, reach * 0.97))
            offset.normalize()
            axis = offset
            target = shoulder + axis * distance
            self.wrist_targets[key] = (Vec3(target), now)
            pole = Vec3(0.35 if side == "Left" else -0.35, 0, -1) + front * 0.45
            elbow = points[elbow_index]
            if (not contact and elbow[3] >= 0.6 and 0 <= elbow[0] < self.width
                    and 0 <= elbow[1] < self.height):
                hint = (elbow[:2] - points[shoulder_index, :2]) * scale
                pole = Vec3(float(hint[0]), 0, float(-hint[1]))
                if near_torso:
                    pole += front * 0.45
            pole -= axis * pole.dot(axis)
            if pole.lengthSquared() < 1e-6:
                pole = axis.cross(Vec3(0, 1, 0))
            if pole.lengthSquared() < 1e-6:
                continue
            pole.normalize()
            old_pole = self.elbow_poles.get(key)
            if old_pole is not None and now - old_pole[1] < 0.4:
                projected = old_pole[0] - axis * old_pole[0].dot(axis)
                if projected.lengthSquared() > 1e-6:
                    projected.normalize()
                    if projected.dot(pole) > math.cos(math.radians(4)):
                        pole = projected
                    else:
                        blend = 1 - math.exp(-max(0.001, now - old_pole[1]) / 0.12)
                        candidate = projected * (1 - blend) + pole * blend
                        if candidate.lengthSquared() > 1e-6:
                            candidate.normalize()
                            pole = candidate
            self.elbow_poles[key] = (Vec3(pole), now)
            along = (upper * upper - lower * lower + distance * distance) / (2 * distance)
            bend = math.sqrt(max(0, upper * upper - along * along))
            elbow_target = shoulder + axis * along + pole * bend
            self._aim(side + "Arm", side + "ForeArm", elbow_target - shoulder)
            actual_elbow = self.controls[side + "ForeArm"].getPos(self.rig)
            self._aim(side + "ForeArm", side + "Hand", target - actual_elbow)
            self.positioned_arms.update((side + "Arm", side + "ForeArm"))

    def _update_hands(self, hands, hand_world, now, contact=False):
        for side, key in (("Left", "left_hand"), ("Right", "right_hand")):
            hand = (hands or {}).get(key)
            if hand is None:
                if now - self.hand_seen.get(key, -math.inf) < 0.18:
                    for name, rotation in self.hand_targets.get(key, {}).items():
                        self.controls[name].setQuat(rotation)
                else:
                    # Missing measurements must not animate fingers from the
                    # idle clip as if they were live detected gestures.
                    for name, control in self.controls.items():
                        if name.startswith(side + "Hand") and name != side + "Hand":
                            control.setTransform(self.rest[name])
                continue
            hand = np.asarray(hand, dtype=float)
            if hand.shape != (21, 3) or not np.isfinite(hand).all():
                continue
            if (contact and key in self.hand_rotations
                    and now - self.hand_seen.get(key, -math.inf) < 0.45):
                # Preserve orientation in rig space as the forearm moves.
                # Reusing a local wrist rotation would rotate/invert the palm
                # with the changing elbow, even when tracking is held.
                for name, rotation in self.hand_targets.get(key, {}).items():
                    self.controls[name].setQuat(rotation)
                self.controls[side + "Hand"].setQuat(self.rig, self.hand_rotations[key])
                self.hand_seen[key] = now
                continue
            world = np.asarray((hand_world or {}).get(key, ()), dtype=float)
            if world.shape == (21, 3) and np.isfinite(world).all():
                # Metric hand coordinates avoid mixing pixel distances with
                # normalized depth when turning the wrist or curling fingers.
                positions = world[:, [0, 2, 1]].copy()
                positions[:, 2] *= -1
            else:
                positions = hand[:, [0, 2, 1]].copy()
                positions[:, 1] *= self.width
                positions[:, 2] *= -1
            # A tracked palm starts from the asset's bind pose. Idle finger
            # curls must not be added to the user's gesture.
            names = [name for name in self.controls if name.startswith(side + "Hand")]
            for name in names:
                self.controls[name].setTransform(self.rest[name])
            self._orient_palm(side, positions)
            for finger, start in (("Thumb", 1), ("Index", 5), ("Middle", 9),
                                  ("Ring", 13), ("Pinky", 17)):
                for segment in range(3):
                    name = f"{side}Hand{finger}{segment + 1}"
                    child = f"{side}Hand{finger}{segment + 2}"
                    self._aim(name, child, positions[start + segment + 1]
                              - positions[start + segment])
            self.hand_seen[key] = now
            self.hand_targets[key] = {name: Quat(self.controls[name].getQuat()) for name in names}
            self.hand_rotations[key] = Quat(self.controls[side + "Hand"].getQuat(self.rig))

    def _update_face(self, face, expressions, alpha):
        values = dict((face or {}).get("blendshapes", {}))
        # These metrics already contain the trained QLoRA HDTF prediction.
        values["jawOpen"] = float(expressions.get("mouth_open_ratio", 0))
        stretch = float(expressions.get("lip_stretch", 0))
        for name in ("mouthStretchLeft", "mouthStretchRight"):
            values[name] = max(float(values.get(name, 0)), stretch * 0.65)
        blink = float(expressions.get("eye_blink_ratio", 0))
        for name in ("eyeBlinkLeft", "eyeBlinkRight"):
            values[name] = 0.75 * float(values.get(name, blink)) + 0.25 * blink
        gaze = np.asarray((face or {}).get("eye_gaze", ()), dtype=float)
        if gaze.shape == (2, 2) and np.isfinite(gaze).all():
            for i, side in enumerate(("Left", "Right")):
                x, y = np.clip(gaze[i], -1, 1)
                inward = x if side == "Left" else -x
                for suffix, weight in (("In", max(0, inward)), ("Out", max(0, -inward)),
                                       ("Up", max(0, y)), ("Down", max(0, -y))):
                    values.setdefault(f"eyeLook{suffix}{side}", float(weight))
        for name, slider in self.sliders.items():
            if slider is None:
                continue
            target = float(np.clip(values.get(name, 0.0), 0, 1))
            if not np.isfinite(target):
                target = 0.0
            old = self.slider_values.get(name, 0.0)
            value = old + (target - old) * min(1, alpha * 1.7)
            self.slider_values[name] = value
            slider.applyFreezeScalar(value)

        target = np.zeros(3)
        matrix = (face or {}).get("facial_pose_matrix")
        if matrix is not None:
            matrix = np.asarray(matrix, dtype=float)
            if matrix.shape == (4, 4) and np.isfinite(matrix).all():
                u, _, vh = np.linalg.svd(matrix[:3, :3])
                rotation = u @ vh
                if np.linalg.det(rotation) > 0:
                    angles = np.asarray(cv2.RQDecomp3x3(rotation)[0])
                    if self.face_reference is None:
                        self.face_reference = angles.copy()
                    target = (angles - self.face_reference + 180) % 360 - 180
        elif face:
            target[2] = expressions.get("head_roll_deg", 0)
        target = np.clip(target, [-25, -45, -35], [25, 45, 35])
        self.face_angles += (target - self.face_angles) * alpha
        pitch, yaw, roll = self.face_angles
        for name, fraction in (("Neck", 0.30), ("Head", 0.70)):
            if name in self.controls:
                control = self.controls[name]
                if face:
                    # Idle head turns must not be added to the user's head pose.
                    control.setTransform(self.rest[name])
                delta = Quat()
                delta.setHpr(Vec3(-yaw, pitch, -roll) * fraction)
                control.setQuat(control.getQuat() * delta)

    def render(self, landmarks=None, hands=None, expressions=None, face=None, hand_world=None,
               body_tracking=None):
        now = time.monotonic()
        dt = min(0.15, max(1 / 120, now - self.last_time))
        self.last_time = now
        alpha = 1 - math.exp(-dt / 0.065)
        if self.animation:
            frames = self.idle.getNumFrames(self.animation)
            rate = self.idle.getFrameRate(self.animation)
            self.idle.pose(self.animation, int((now - self.started) * rate) % frames)
            self.idle.update(force=True)
        for name, control in self.controls.items():
            joint = self.idle_joints[name]
            control.setMat(joint.getValue() if joint is not None else self.rest[name].getMat())
        self._update_posture(body_tracking)
        self._update_body(landmarks, body_tracking)
        self.positioned_arms.clear()
        self._position_hands(landmarks, hands,
                             contact=(body_tracking or {}).get("hands_contact", False))
        contact = (body_tracking or {}).get("hands_contact", False)
        self._update_hands(hands, hand_world, now,
                           contact=contact)
        self._update_face(face, expressions or {}, alpha)
        for name, control in self.controls.items():
            target = control.getQuat()
            previous = self.previous_quats.get(name, target)
            # Wrist targets already have temporal smoothing. Blending the two
            # arm joints independently breaks the IK clearance constraints.
            held_palm = (contact and name in ("LeftHand", "RightHand")
                         and name.replace("Hand", "Arm") in self.positioned_arms)
            current = (target if name in self.positioned_arms or held_palm
                       else self._blend_quat(previous, target, alpha))
            control.setQuat(current)
            self.previous_quats[name] = Quat(current)
        if contact:
            # Spine/shoulder smoothing above changes the palm's parent frame.
            # Reapply the held rig-space orientation after those transforms.
            for side, key in (("Left", "left_hand"), ("Right", "right_hand")):
                if (side + "Arm" in self.positioned_arms
                        and key in self.hand_rotations
                        and now - self.hand_seen.get(key, -math.inf) < 0.45):
                    control = self.controls[side + "Hand"]
                    control.setQuat(self.rig, self.hand_rotations[key])
                    self.previous_quats[side + "Hand"] = Quat(control.getQuat())
        if body_tracking is not None:
            self._ground_body(dt)
        self.bundle.forceUpdate()
        self.actor.update(force=True)
        self._update_camera(face, dt, body_tracking)
        self.base.graphicsEngine.renderFrame()
        raw = self.texture.getRamImageAs("BGR")
        width, height = self.texture.getXSize(), self.texture.getYSize()
        if not raw or width != self.width or height != self.height:
            raise RuntimeError("Native avatar renderer did not return a complete frame.")
        return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)[::-1].copy()

    def close(self):
        for name in ("actor", "idle"):
            actor = getattr(self, name, None)
            if actor is not None:
                actor.cleanup()
        if self.base is not None:
            self.base.destroy()
            self.base = None
