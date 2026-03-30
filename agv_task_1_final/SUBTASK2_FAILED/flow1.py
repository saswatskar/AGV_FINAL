# flow1.py
import math
import time
from typing import Optional

import cv2
import numpy as np
import pybullet as p

from lucas_kanade import LucasKanadeTracker


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _as_2d(a):
    return np.asarray(a, dtype=np.float32).reshape(-1, 2)


class FlowController:
    """
    Sparse optical-flow controller for AGV Subtask 2.

    Main ideas:
      - track Shi-Tomasi features with Lucas-Kanade
      - estimate FOE from flow lines
      - estimate TTC for each feature
      - build a smoothed obstacle risk field
      - steer away from obstacle imbalance
      - keep a mild forward/center bias so the car does not drift off-road

    This version is tuned for the provided PyBullet road corridor.
    """

    def __init__(
        self,
        car_id,
        steering_joints,
        motor_joints,
        image_size=(640, 480),
        gui=True,
        tracker=None,
    ):
        self.car_id = car_id
        self.steering_joints = steering_joints
        self.motor_joints = motor_joints
        self.w, self.h = image_size
        self.gui = gui

        self.tracker = tracker or LucasKanadeTracker(
            win=31,
            levels=3,
            max_iter=20,
            eps=0.03,
            min_eig=1e-3,
        )

        # Camera
        self.fov = 60.0
        self.near = 0.1
        self.far = 80.0
        self.cam_forward = 0.40
        self.cam_height = 0.55
        self.look_ahead = 10.0

        # Feature detection
        self.max_corners = 600
        self.quality_level = 0.01
        self.min_distance = 7
        self.block_size = 7

        # Optical flow / TTC
        self.min_flow_mag = 0.60
        self.ttc_floor = 0.15
        self.ttc_ceiling = 6.0
        self.min_tracks = 35

        # Potential-field style gains
        self.alpha_att = 0.018
        self.gamma_rep = 2.20
        self.gaussian_sigma = 11.0
        self.gaussian_ksize = (0, 0)

        # Steering control gains
        self.max_steer_deg = 40.0
        self.max_steer_rad = math.radians(self.max_steer_deg)

        # IMPORTANT: keep this at -1.0 for this PyBullet racecar setup.
        self.steer_sign = -1.0

        self.base_speed = 3.25
        self.min_speed = 1.10
        self.max_speed = 5.50
        self.k_speed = 0.45

        self.steer_smooth = 0.82
        self.throttle_smooth = 0.75

        # State
        self.prev_gray = None
        self.prev_pts = None
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.frame_idx = 0

        # Sparse feature ROI: lower road corridor, not sky or far horizon
        self.roi_mask = self._make_roi_mask()

        # Goal point in image space (just a forward attractor)
        self.goal_px = np.array([self.w * 0.50, self.h * 0.22], dtype=np.float32)
        self.vehicle_px = np.array([self.w * 0.50, self.h * 0.86], dtype=np.float32)

    def _make_roi_mask(self):
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        pts = np.array(
            [
                [int(self.w * 0.10), self.h - 1],
                [int(self.w * 0.90), self.h - 1],
                [int(self.w * 0.68), int(self.h * 0.30)],
                [int(self.w * 0.32), int(self.h * 0.30)],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(mask, pts, 255)
        return mask

    def _camera_mats(self):
        pos, orn = p.getBasePositionAndOrientation(self.car_id)
        rot = np.array(p.getMatrixFromQuaternion(orn), dtype=np.float32).reshape(3, 3)

        forward = rot @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
        up = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float32)

        cam_pos = np.array(pos, dtype=np.float32) + forward * self.cam_forward + up * self.cam_height
        target = cam_pos + forward * self.look_ahead + np.array([0.0, 0.0, -0.18], dtype=np.float32)

        view = p.computeViewMatrix(
            cameraEyePosition=cam_pos.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
        )
        proj = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=float(self.w) / float(self.h),
            nearVal=self.near,
            farVal=self.far,
        )
        return view, proj

    def grab_frame(self):
        view, proj = self._camera_mats()
        renderer = p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER

        _, _, rgba, _, _ = p.getCameraImage(
            width=self.w,
            height=self.h,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=renderer,
        )

        rgba = np.reshape(np.array(rgba, dtype=np.uint8), (self.h, self.w, 4))
        rgb = rgba[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return gray

    def detect_features(self, gray):
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
            mask=self.roi_mask,
        )
        if pts is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return pts.astype(np.float32)

    def estimate_foe(self, pts, flows):
        pts = _as_2d(pts)
        flows = _as_2d(flows)

        if len(pts) < 6:
            return np.array([self.w * 0.5, self.h * 0.30], dtype=np.float32)

        x = pts[:, 0]
        y = pts[:, 1]
        vx = flows[:, 0]
        vy = flows[:, 1]

        # vy * X - vx * Y = x * vy - y * vx
        A = np.column_stack([vy, -vx]).astype(np.float32)
        b = (x * vy - y * vx).astype(np.float32)

        mag = np.linalg.norm(flows, axis=1)
        w = np.clip(mag / (np.median(mag) + 1e-6), 0.5, 3.0).astype(np.float32)
        Aw = A * w[:, None]
        bw = b * w

        try:
            foe, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)
            foe = foe.astype(np.float32)
        except np.linalg.LinAlgError:
            foe = np.array([self.w * 0.5, self.h * 0.30], dtype=np.float32)

        if not np.all(np.isfinite(foe)):
            foe = np.array([self.w * 0.5, self.h * 0.30], dtype=np.float32)

        return foe

    def build_obstacle_force(self, pts, flows, ttc):
        """
        Paper-inspired obstacle field:
          risk = 1 / TTC
          then Gaussian smoothing
          then gradient-like repulsion

        Returns:
          frep_img: repulsive force in image coordinates
          risk_strength: scalar in [0,1]
        """
        pts = _as_2d(pts)
        flows = _as_2d(flows)
        ttc = np.asarray(ttc, dtype=np.float32).reshape(-1)

        if len(pts) == 0:
            return np.zeros(2, dtype=np.float32), 0.0

        n = min(len(pts), len(flows), len(ttc))
        pts = pts[:n]
        flows = flows[:n]
        ttc = ttc[:n]

        # Stable TTC weighting: never let it explode
        risk = 1.0 / (ttc + 1.0)
        risk = np.clip(risk, 0.0, 1.0)

        # Suppress very weak signals
        risk_u8 = (risk * 255.0).astype(np.uint8).reshape(-1, 1)

        if len(np.unique(risk_u8)) > 1:
            _, thr = cv2.threshold(risk_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            obs_idx = (risk_u8.flatten() >= thr).astype(bool)
        else:
            obs_idx = (risk > np.mean(risk)).astype(bool)

        obs_idx = np.asarray(obs_idx, dtype=bool).reshape(-1)
        if obs_idx.shape[0] != pts.shape[0]:
            obs_idx = obs_idx[: pts.shape[0]]

        obs_pts = pts[obs_idx]
        obs_risk = risk[obs_idx]

        canvas = np.zeros((self.h, self.w), dtype=np.float32)

        for (x, y), r in zip(obs_pts, obs_risk):
            xi = int(round(float(x)))
            yi = int(round(float(y)))
            if 0 <= xi < self.w and 0 <= yi < self.h:
                cv2.circle(canvas, (xi, yi), 8, float(r), -1)

        smooth = cv2.GaussianBlur(
            canvas,
            ksize=self.gaussian_ksize,
            sigmaX=self.gaussian_sigma,
            sigmaY=self.gaussian_sigma,
            borderType=cv2.BORDER_DEFAULT,
        )

        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)

        ys, xs = np.where(canvas > 0.0)
        if len(xs) == 0:
            return np.zeros(2, dtype=np.float32), float(np.mean(risk))

        grads = np.stack([gx[ys, xs], gy[ys, xs]], axis=1)
        weights = smooth[ys, xs] + 1e-6

        frep = self.gamma_rep * np.sum(grads * weights[:, None], axis=0) / (len(xs) + 1e-6)
        risk_strength = float(np.clip(np.max(risk), 0.0, 1.0))
        return frep.astype(np.float32), risk_strength

    def build_attractive_force(self):
        delta = self.goal_px - self.vehicle_px
        dist = float(np.linalg.norm(delta) + 1e-6)
        direction = delta / dist
        mag = self.alpha_att * min(dist, 250.0)
        return (mag * direction).astype(np.float32)

    def steering_from_scene(self, pts, ttc, foe):
        """
        Steering law:
          - left/right obstacle imbalance
          - mild FOE stabilizer
          - weak center-preserving bias

        This avoids the unstable road-segmentation step.
        """
        pts = _as_2d(pts)
        ttc = np.asarray(ttc, dtype=np.float32).reshape(-1)

        if len(pts) < 10:
            return 0.0, 1.0, 0.0, 0.0

        n = min(len(pts), len(ttc))
        pts = pts[:n]
        ttc = ttc[:n]

        # Stable weighting: close = more important, but not explosive
        w = 1.0 / (ttc + 1.0)

        # Use only lower / nearer region where road obstacles live
        valid = pts[:, 1] > self.h * 0.30
        pts = pts[valid]
        w = w[valid]

        if len(pts) < 10:
            return 0.0, 1.0, 0.0, 0.0

        # Ignore the extreme edges to reduce lane-edge confusion
        margin = int(self.w * 0.12)
        valid_x = (pts[:, 0] > margin) & (pts[:, 0] < self.w - margin)
        pts = pts[valid_x]
        w = w[valid_x]

        if len(pts) < 10:
            return 0.0, 1.0, 0.0, 0.0

        cx = self.w * 0.5

        left_w = float(w[pts[:, 0] < cx - 8].sum())
        right_w = float(w[pts[:, 0] > cx + 8].sum())
        mid_w = float(w[(pts[:, 0] >= cx - 8) & (pts[:, 0] <= cx + 8)].sum())
        total = left_w + right_w + mid_w + 1e-6

        # Positive means right side is more dangerous.
        lateral_bias = (right_w - left_w) / total

        foe_bias = 0.0
        if foe is not None and np.all(np.isfinite(foe)):
            foe_bias = float((foe[0] - cx) / (cx + 1e-6))

        # Weak center bias to keep the car from slowly drifting off road.
        center_bias = -0.18 * foe_bias

        # Obstacles decide the main direction; center bias only stabilizes.
        steer_raw = -(2.15 * lateral_bias) + center_bias

        steer_raw = float(np.clip(steer_raw, -1.0, 1.0))

        # Never let speed collapse to zero; the wall/end obstacle should be handled by steering first.
        speed_scale = 1.0 - 0.50 * abs(steer_raw)
        speed_scale = float(np.clip(speed_scale, 0.45, 1.0))

        return steer_raw, speed_scale, lateral_bias, foe_bias

    def apply(self, steer_norm, throttle_norm):
        steer_norm = clamp(float(steer_norm), -1.0, 1.0)
        throttle_norm = clamp(float(throttle_norm), -1.0, 1.0)

        steer_angle = self.steer_sign * steer_norm * self.max_steer_rad

        for j in self.steering_joints:
            p.setJointMotorControl2(
                bodyUniqueId=self.car_id,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=steer_angle,
                force=25.0,
            )

        wheel_speed = throttle_norm * 35.0
        for j in self.motor_joints:
            p.setJointMotorControl2(
                bodyUniqueId=self.car_id,
                jointIndex=j,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=wheel_speed,
                force=30.0,
            )

    def step(self):
        gray = self.grab_frame()
        self.frame_idx += 1

        if self.prev_gray is None or self.prev_pts is None:
            self.prev_gray = gray
            self.prev_pts = self.detect_features(gray)
            return 0.0, 0.0, {
                "foe": None,
                "track_count": 0,
                "risk": 0.0,
                "F_att": np.zeros(2, dtype=np.float32),
                "F_rep": np.zeros(2, dtype=np.float32),
                "F_total": np.zeros(2, dtype=np.float32),
                "steer": 0.0,
                "throttle": 0.0,
                "speed": 0.0,
                "v_ref": 0.0,
                "lateral_bias": 0.0,
                "foe_bias": 0.0,
            }

        next_pts, status = self.tracker.track(self.prev_gray, gray, self.prev_pts)
        status = status.reshape(-1).astype(bool)

        old_pts = self.prev_pts.reshape(-1, 2)[status]
        new_pts = next_pts.reshape(-1, 2)[status]

        if len(new_pts) == 0:
            self.prev_gray = gray
            self.prev_pts = self.detect_features(gray)
            return 0.0, 0.15, {
                "foe": None,
                "track_count": 0,
                "risk": 0.0,
                "F_att": np.zeros(2, dtype=np.float32),
                "F_rep": np.zeros(2, dtype=np.float32),
                "F_total": np.zeros(2, dtype=np.float32),
                "steer": 0.0,
                "throttle": 0.15,
                "speed": 0.0,
                "v_ref": 0.0,
                "lateral_bias": 0.0,
                "foe_bias": 0.0,
            }

        flows = new_pts - old_pts
        mags = np.linalg.norm(flows, axis=1)

        valid = np.isfinite(mags) & (mags > self.min_flow_mag)
        new_pts = new_pts[valid].reshape(-1, 2)
        flows = flows[valid].reshape(-1, 2)
        mags = mags[valid].reshape(-1)

        if len(new_pts) < self.min_tracks:
            self.prev_gray = gray
            self.prev_pts = self.detect_features(gray)
            return 0.0, 0.15, {
                "foe": None,
                "track_count": int(len(new_pts)),
                "risk": 0.0,
                "F_att": np.zeros(2, dtype=np.float32),
                "F_rep": np.zeros(2, dtype=np.float32),
                "F_total": np.zeros(2, dtype=np.float32),
                "steer": 0.0,
                "throttle": 0.15,
                "speed": 0.0,
                "v_ref": 0.0,
                "lateral_bias": 0.0,
                "foe_bias": 0.0,
            }

        foe = self.estimate_foe(new_pts, flows)

        dist = np.linalg.norm(new_pts - foe[None, :], axis=1)
        ttc = dist / (mags + 1e-6)
        ttc = ttc.reshape(-1)

        frep, risk_strength = self.build_obstacle_force(new_pts, flows, ttc)
        fatt = self.build_attractive_force()
        f_total = fatt - frep

        # Vehicle forward speed
        _, orn = p.getBasePositionAndOrientation(self.car_id)
        rot = np.array(p.getMatrixFromQuaternion(orn), dtype=np.float32).reshape(3, 3)
        forward_vec = rot @ np.array([1.0, 0.0, 0.0], dtype=np.float32)

        lin_vel, _ = p.getBaseVelocity(self.car_id)
        lin_vel = np.array(lin_vel, dtype=np.float32)
        speed = float(abs(np.dot(lin_vel, forward_vec)))

        steer_raw, speed_scale, lateral_bias, foe_bias = self.steering_from_scene(
            new_pts, ttc, foe
        )

        # Reduce speed when obstacle risk increases or steering demand grows,
        # but keep a floor so the vehicle does not stall early.
        v_ref = clamp(self.base_speed * speed_scale, self.min_speed, self.max_speed)

        # If a strong obstacle is directly ahead, reduce speed further, but not to zero.
        front_band = new_pts[:, 1] > self.h * 0.52
        center_band = np.abs(new_pts[:, 0] - self.w * 0.5) < self.w * 0.12
        head_on_risk = float(np.clip(np.mean((1.0 / (ttc + 1.0))[front_band & center_band]) if np.any(front_band & center_band) else 0.0, 0.0, 1.0))
        v_ref *= float(np.clip(1.0 - 0.35 * head_on_risk, 0.65, 1.0))

        throttle = self.k_speed * (v_ref - speed)
        throttle = clamp(throttle, -1.0, 1.0)

        # Smoothing
        steer = self.steer_smooth * self.prev_steer + (1.0 - self.steer_smooth) * steer_raw
        throttle = self.throttle_smooth * self.prev_throttle + (1.0 - self.throttle_smooth) * throttle

        self.prev_steer = steer
        self.prev_throttle = throttle

        self.prev_gray = gray
        self.prev_pts = new_pts.reshape(-1, 1, 2).astype(np.float32)

        debug = {
            "foe": foe,
            "track_count": int(len(new_pts)),
            "risk": float(risk_strength),
            "F_att": fatt,
            "F_rep": frep,
            "F_total": f_total,
            "steer": float(steer),
            "throttle": float(throttle),
            "speed": float(speed),
            "v_ref": float(v_ref),
            "lateral_bias": float(lateral_bias),
            "foe_bias": float(foe_bias),
            "head_on_risk": float(head_on_risk),
        }
        return steer, throttle, debug

    def run(self, dt=1.0 / 60.0, stop_x=31.0, print_every=30):
        print("[FlowController] starting...")
        while True:
            steer, throttle, debug = self.step()
            self.apply(steer, throttle)
            p.stepSimulation()
            time.sleep(dt)

            if self.frame_idx % print_every == 0:
                print(
                    f"frame={self.frame_idx:04d} "
                    f"steer={debug['steer']:+.3f} "
                    f"throttle={debug['throttle']:+.3f} "
                    f"tracks={debug['track_count']:03d} "
                    f"risk={debug['risk']:.3f} "
                    f"lat_bias={debug['lateral_bias']:+.3f} "
                    f"foe_bias={debug['foe_bias']:+.3f} "
                    f"head_on={debug['head_on_risk']:.3f} "
                    f"speed={debug['speed']:.2f} "
                    f"v_ref={debug['v_ref']:.2f}"
                )

            pos, _ = p.getBasePositionAndOrientation(self.car_id)
            if pos[0] >= stop_x:
                print(f"[FlowController] goal reached at x={pos[0]:.2f}")
                break