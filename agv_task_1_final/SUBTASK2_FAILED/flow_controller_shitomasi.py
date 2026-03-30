"""
flow_controller.py
──────────────────────────────────────────────────────────────────────────────
Visual Potential Field controller for autonomous driving in PyBullet.

Based on:
  Capito, Ozguner, Redmill — "Optical Flow based Visual Potential Field
  for Autonomous Driving" (IV 2020)

  Sazbon, Rotstein, Rivlin — "Finding the Focus of Expansion and
  Estimating Range using Optical Flow" (MVA 2004)

Feature strategy : Shi-Tomasi corner detection with periodic re-detection.
Obstacle detect  : TTC-based (Eq. 3) + flow-speed anomaly classification.
Controller       : Gradient Tracking Sliding Mode (GTSMC), Sec. VI.

Full pipeline per frame
───────────────────────
  1.  Capture grayscale image from forward-mounted camera
  2.  Track features: Shi-Tomasi → pyramidal Lucas-Kanade (cv2 back-end)
  3.  Compute FOE via least-squares (Sec. III-A)
  4.  Compute TTC for each tracked point (Eq. 3)
  5.  Classify obstacle points (flow-speed anomaly + radial-flow test)
  6.  Compute Visual Potential Field
        a. Target / Attractive field  (Sec. IV-A, Eq. 7-8)
        b. Obstacle repulsive field   (Sec. IV-B, Eq. 9)
        c. Road boundary Morse field  (Sec. IV-C, Eq. 10-11, 18)
  7.  Sum forces → desired heading ψd (Eq. 27)
  8.  GTSMC: sliding manifold sr → steering angle (Eq. 28-30)

Key fixes over the original grid controller
────────────────────────────────────────────
  • Shi-Tomasi features (textured obstacles tracked reliably) instead of a
    fixed pixel grid that drifts on low-texture regions.
  • FOE least-squares formulation follows the paper exactly (Sec. III-A)
    rather than the approximate version in the original.
  • Road Morse potential derived correctly for the narrower simulation road
    (ROAD_HALF_WIDTH tuned to match PyBullet road half-extents = 1.16 m,
     with the car spawn centred at y = 0).
  • GTSMC controller uses a soft-sign (tanh) to suppress chattering and
    integrates the steering angle with clamping, matching Eq. 29-30.
  • Feature re-detection falls back gracefully; never crashes on empty sets.
"""

import cv2
import numpy as np
import pybullet as p
import time

from lucas_kanade import LucasKanadeTracker_1
from simulation_setup import setup_simulation


# ── Colour palette ─────────────────────────────────────────────────────────
CLR_ATT      = (0,   220,   0)   # green      – attractive force arrow
CLR_REP      = (0,    60, 255)   # red        – repulsive  force arrow
CLR_TOT      = (0,   220, 255)   # yellow     – total      force arrow
CLR_FOE      = (255,   0,   0)   # blue       – FOE marker
CLR_FLOW_BG  = (0,   180,   0)   # dim-green  – background flow trail
CLR_FLOW_OBS = (0,     0, 220)   # red        – obstacle   flow trail
CLR_OBS_RING = (0,     0, 255)   # bright-red – obstacle ring


# ── Shi-Tomasi / feature parameters ────────────────────────────────────────
FEATURE_PARAMS = dict(
    maxCorners   = 200,
    qualityLevel = 0.01,
    minDistance  = 10,
    blockSize    = 7,
)
REDETECT_EVERY = 20   # re-run Shi-Tomasi every N frames
MIN_FEATURES   = 20   # force re-detect if fewer points survive tracking


# ── Obstacle-detection thresholds ──────────────────────────────────────────
FLOW_ANOMALY_K    = 1.6   # speed > K × median  →  candidate obstacle
MIN_OBS_SPEED     = 0.3   # absolute floor (px/frame)
TTC_THRESHOLD     = 20.0  # only react if TTC < this value (frames)
RADIAL_DOT_THRESH = 0.3   # cosine threshold; flow more radial than this
                           # is treated as background expansion, not obstacle


# ── Road potential field (Morse, Table I — Capito 2020) ────────────────────
#   Road half-extents in simulation = 1.16 m (create_road_and_obstacles).
#   Car starts at y = 0; obstacle slalom at y = ±0.38 m.
#   Warn boundary set inside the hard road edge.
ROAD_HALF_WIDTH = 0.55   # |y| beyond which Morse potential fires (m)
A_MORSE         = 0.5    # Morse depth  A  (Table I)
B_MORSE         = 8.0    # Morse variance parameter b  (tuned for sim scale)


# ── Force weights / gains ───────────────────────────────────────────────────
ALPHA_ATT  = 1.0       # attractive force scaling  α  (Eq. 8)
GAMMA_OBS  = 10000.0   # repulsive obstacle gain   γ  (Eq. 9)
LAMBDA_X   = 1.0       # longitudinal force weight λX (Eq. 19)
LAMBDA_Y   = 1.0       # lateral     force weight  λY (Eq. 20)


# ── GTSMC parameters ────────────────────────────────────────────────────────
CR          = 0.8   # rotational-manifold gain cr   (Eq. 28)
U0          = 0.8   # control amplitude u0          (Eq. 30)
STEER_LIMIT = 0.6   # hard clamp on steering (rad)


# ──────────────────────────────────────────────────────────────────────────
#  HUD helpers
# ──────────────────────────────────────────────────────────────────────────

def _draw_force_arrow(img, origin, force_x, scale=60.0,
                      color=(255, 255, 255), label="",
                      thickness=2, tip_length=0.35):
    ox, oy = int(origin[0]), int(origin[1])
    ex     = int(ox + force_x * scale)
    cv2.arrowedLine(img, (ox, oy), (ex, oy), color,
                    thickness, tipLength=tip_length)
    if label:
        cv2.putText(img, f"{label}: {force_x:+.2f}",
                    (ox + 4, oy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def _draw_force_panel(rgb, f_att, f_rep, f_total, H, W):
    panel_h = 72
    y0      = H - panel_h
    overlay = rgb.copy()
    cv2.rectangle(overlay, (0, y0), (W, H), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, rgb, 0.45, 0, rgb)

    cx      = W // 2
    row_gap = 22
    cv2.line(rgb, (cx, y0 + 2), (cx, H - 2), (80, 80, 80), 1)

    rows   = [y0 + 10, y0 + 10 + row_gap, y0 + 10 + 2 * row_gap]
    forces = [f_att,   f_rep,              f_total]
    labels = ["F_att", "F_rep",            "F_tot"]
    colors = [CLR_ATT, CLR_REP,            CLR_TOT]

    for ry, fx, lbl, clr in zip(rows, forces, labels, colors):
        _draw_force_arrow(rgb, (cx, ry), fx,
                          scale=60.0, color=clr, label=lbl)


# ──────────────────────────────────────────────────────────────────────────
#  FlowController
# ──────────────────────────────────────────────────────────────────────────

class FlowController:
    """
    Optical-flow Visual Potential Field controller.

    Implements the complete pipeline from Capito et al. (IV 2020):

      Feature tracking  — Shi-Tomasi + pyramidal LK (via LucasKanadeTracker_1)
      FOE estimation    — Least-squares on flow vectors (Sec. III-A)
      TTC               — Per-pixel time-to-contact (Eq. 3)
      Potential Fields  — Attractive + Obstacle repulsive + Road Morse
      Controller        — GTSMC lateral steering (Sec. VI, Eq. 27-30)
    """

    def __init__(self, car_id, steer_joints, motor_joints):
        self.car_id       = car_id
        self.steer_joints = steer_joints
        self.motor_joints = motor_joints

        # LK tracker (window 25 px, 3 pyramid levels — matches Capito Table II)
        self.tracker   = LucasKanadeTracker_1(win=25, levels=3,
                                               max_iter=20, eps=0.03)
        self.prev_gray = None
        self.p0        = None          # (N,1,2) float32 current feature set

        self.W, self.H = 320, 240      # camera resolution
        self.frame_idx = 0

        # GTSMC integration state
        self._psi_d_prev  = 0.0        # previous desired heading (for ψ̇d)
        self._steer_angle = 0.0        # integrated steering angle (Eq. 29)
        self._dt          = 1.0 / 60.0 # simulation timestep

    # ── Camera ─────────────────────────────────────────────────────────────

    def _get_frame(self):
        """
        Render a forward-facing camera image attached to the car.
        Returns (gray HxW uint8, rgb HxWx3 uint8).
        """
        pos, orn   = p.getBasePositionAndOrientation(self.car_id)
        R          = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        cam_pos    = np.array(pos) + R @ [0.4, 0.0, 0.35]
        target_pos = np.array(pos) + R @ [2.0, 0.0, 0.35]

        view = p.computeViewMatrix(
            cam_pos.tolist(), target_pos.tolist(), [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(
            fov=90, aspect=self.W / self.H, nearVal=0.1, farVal=100)

        img_arr = p.getCameraImage(self.W, self.H, view, proj,
                                   renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb  = img_arr[2][:, :, :3]                       # HxWx3 RGB uint8
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)      # HxW   uint8
        return gray, rgb.copy()

    # ── Feature management ─────────────────────────────────────────────────

    def _detect(self, gray):
        """Run Shi-Tomasi corner detection; returns (N,1,2) float32."""
        pts = cv2.goodFeaturesToTrack(gray, mask=None, **FEATURE_PARAMS)
        if pts is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return pts.astype(np.float32)

    # ── FOE — Sec. III-A (Capito 2020) ─────────────────────────────────────

    def _compute_foe(self, p0, p1):
        """
        Least-squares FOE from a set of tracked correspondences.

        For each point i with position (x,y) and flow (vx,vy):
            a_i0 = vy,   a_i1 = -vx,   b_i = x·vy − y·vx
        then  FOE = (AᵀA)⁻¹ Aᵀb

        Falls back to image centre when the system is under-constrained.
        """
        v  = p1 - p0
        x, y   = p0[:, 0], p0[:, 1]
        vx, vy = v[:, 0],  v[:, 1]

        A   = np.column_stack((vy, -vx))   # (N,2)
        b   = x * vy - y * vx              # (N,)
        ATA = A.T @ A                      # (2,2)

        try:
            det = ATA[0, 0] * ATA[1, 1] - ATA[0, 1] * ATA[1, 0]
            if abs(det) < 1e-6:
                return np.array([self.W / 2.0, self.H / 2.0])
            foe = np.linalg.solve(ATA, A.T @ b)
            # Allow the FOE to wander somewhat outside the frame
            foe[0] = np.clip(foe[0], -self.W,     2 * self.W)
            foe[1] = np.clip(foe[1], -self.H,     2 * self.H)
            return foe
        except np.linalg.LinAlgError:
            return np.array([self.W / 2.0, self.H / 2.0])

    # ── TTC — Eq. 3 (Capito 2020) ──────────────────────────────────────────

    @staticmethod
    def _ttc(x, y, vx, vy, foe):
        """
        Time-to-contact for each tracked point (Eq. 3):
            TTC_i = ‖p_i − FOE‖ / ‖v_i‖
        """
        dist = np.sqrt((x - foe[0])**2 + (y - foe[1])**2) + 1e-5
        spd  = np.sqrt(vx**2 + vy**2) + 1e-5
        return dist / spd

    # ── Obstacle classification ─────────────────────────────────────────────

    @staticmethod
    def _is_obstacle(speeds):
        """
        Flow-speed anomaly: points whose speed significantly exceeds
        the scene median are treated as near/looming obstacles.

        Using the median rather than the mean makes this robust to
        large obstacle clusters (up to ~50% of the tracked point set).
        """
        med = np.median(speeds)
        thr = max(FLOW_ANOMALY_K * med, MIN_OBS_SPEED)
        return speeds > thr

    # ── Road Potential Field — Sec. IV-C, Eq. 10-11, 18 ───────────────────

    def _road_force_y(self, y_car):
        """
        Repulsive lateral force from road boundaries (negative gradient of
        the modified Morse potential, Eq. 18):

            Usr = A (1 − e^{−b(y−yr)})²      (right boundary, yr < 0)
            Usl = A (1 − e^{ b(y−yl)})²      (left  boundary, yl > 0)

        ∂Usr/∂y =  2Ab e^{−b(y−yr)} (1 − e^{−b(y−yr)})
        ∂Usl/∂y = −2Ab e^{ b(y−yl)} (1 − e^{ b(y−yl)})

        F_road_y = −(∂Usr/∂y + ∂Usl/∂y)
        """
        yr = -ROAD_HALF_WIDTH   # right boundary (negative Y in PyBullet)
        yl =  ROAD_HALF_WIDTH   # left  boundary (positive Y)

        dUsr_dy = (2 * A_MORSE * B_MORSE
                   * np.exp(-B_MORSE * (y_car - yr))
                   * (1 - np.exp(-B_MORSE * (y_car - yr))))

        dUsl_dy = (-2 * A_MORSE * B_MORSE
                   * np.exp(B_MORSE * (y_car - yl))
                   * (1 - np.exp(B_MORSE * (y_car - yl))))

        return -(dUsr_dy + dUsl_dy)

    # ── Obstacle Potential Field — Sec. IV-B, Eq. 9 ───────────────────────

    def _obstacle_forces(self, good_old, good_new, speeds, foe, rgb=None):
        """
        Repulsive forces from obstacle-classified tracked points.

        Each point that passes all filters contributes:
            F_obs_Y += sign(lateral) × γ / TTC_i
            F_obs_X -= γ / TTC_i          (longitudinal slow-down)

        Filters applied per point (all must be True to contribute):
            • flow-speed anomaly   (_is_obstacle)
            • looming toward FOE   (dot(r, v) > 0)
            • not pure background  (radial cosine < RADIAL_DOT_THRESH)
            • TTC < TTC_THRESHOLD
            • in lower image half  (y > H × 0.3)

        Parameters
        ----------
        good_old, good_new : (N,2) float32 — previous / current positions
        speeds             : (N,)  float32 — per-point flow magnitudes
        foe                : (2,)  float64 — focus of expansion [px, py]
        rgb                : optional HxWx3 array for debug drawing

        Returns
        -------
        F_obs_X, F_obs_Y : float — obstacle repulsive forces (local frame)
        """
        x, y   = good_old[:, 0], good_old[:, 1]
        vx     = good_new[:, 0] - good_old[:, 0]
        vy     = good_new[:, 1] - good_old[:, 1]

        ttc         = self._ttc(x, y, vx, vy, foe)
        looming     = ((x - foe[0]) * vx + (y - foe[1]) * vy) > 0
        is_obs      = self._is_obstacle(speeds)

        # Radial alignment: purely radially-expanding flow is background
        r_vec  = np.column_stack([x - foe[0], y - foe[1]])
        f_vec  = np.column_stack([vx, vy])
        norm_r = np.linalg.norm(r_vec, axis=1, keepdims=True) + 1e-5
        norm_f = np.linalg.norm(f_vec, axis=1, keepdims=True) + 1e-5
        dot_r  = np.sum((r_vec / norm_r) * (f_vec / norm_f), axis=1)
        not_bg = dot_r < RADIAL_DOT_THRESH

        F_obs_X = 0.0
        F_obs_Y = 0.0

        for i in range(len(good_new)):
            nx, ny = int(good_new[i, 0]), int(good_new[i, 1])
            ox, oy = int(good_old[i, 0]), int(good_old[i, 1])

            # Flow trail visualisation
            if rgb is not None:
                trail_clr = CLR_FLOW_OBS if is_obs[i] else CLR_FLOW_BG
                cv2.line(rgb, (nx, ny), (ox, oy), trail_clr, 1)
                cv2.circle(rgb, (nx, ny), 2, trail_clr, -1)

            if (is_obs[i]
                    and looming[i]
                    and not_bg[i]
                    and ttc[i] < TTC_THRESHOLD
                    and good_old[i, 1] > self.H * 0.3):

                # Lateral direction: obstacle left of FOE → push right (+Y)
                dir_y = 1.0 if x[i] < foe[0] else -1.0
                gain  = GAMMA_OBS / (ttc[i] + 1e-2)

                F_obs_Y += dir_y * gain
                F_obs_X -= gain          # always request deceleration

                if rgb is not None:
                    cv2.circle(rgb, (int(x[i]), int(y[i])), 6, CLR_OBS_RING, 2)

        return F_obs_X, F_obs_Y

    # ── GTSMC steering — Sec. VI, Eq. 27-30 ───────────────────────────────

    def _gtsmc_steer(self, F_total_X, F_total_Y, yaw):
        """
        Gradient Tracking Sliding Mode Controller.

        Desired heading (Eq. 27):
            ψd = atan2(F_Y, F_X)

        Orientation error:
            ψe = ψ − ψd       (wrapped to [−π, π])

        Rotational sliding manifold (Eq. 28):
            sr = cr·ψe + ψ̇e ≈ cr·ψe + ψ̇d

        Control law (Eq. 29-30, with tanh soft-sign for chattering reduction):
            δ̇ = u = −u0·tanh(k·sr)
            δ  = ∫ u dt  (clamped to ±STEER_LIMIT)

        Returns
        -------
        steer_angle : float — target steering angle for PyBullet joints
        psi_d       : float — desired heading (rad) for diagnostics
        """
        psi_d = np.arctan2(F_total_Y, max(abs(F_total_X), 0.1))

        psi_e = yaw - psi_d
        psi_e = (psi_e + np.pi) % (2 * np.pi) - np.pi  # wrap to [−π, π]

        # Approximate ψ̇d via finite difference of desired heading
        psi_d_dot         = (psi_d - self._psi_d_prev) / self._dt
        self._psi_d_prev  = psi_d

        sr = CR * psi_e + psi_d_dot

        # Smooth sign → suppress chattering (standard GTSMC modification)
        u = -U0 * np.tanh(5.0 * sr)

        self._steer_angle += u * self._dt
        self._steer_angle  = np.clip(self._steer_angle,
                                     -STEER_LIMIT, STEER_LIMIT)
        return self._steer_angle, psi_d

    # ── Main navigation loop ───────────────────────────────────────────────

    def navigate(self, target_speed: float = 10.0):
        """
        Run the car from its spawn position to the end wall at x ≈ 31.7 m,
        avoiding slalom obstacles using visual potential fields.

        Parameters
        ----------
        target_speed : float
            Wheel velocity control target (rad/s passed to PyBullet joints).
            Default 10.0 gives a comfortable approach speed.
        """
        print("[FlowController] Starting — press Q to quit")
        self.prev_gray, _ = self._get_frame()
        self.p0           = self._detect(self.prev_gray)

        while True:
            self.frame_idx += 1
            curr_gray, rgb = self._get_frame()

            # ── Drive wheels at constant forward speed ─────────────────────
            p.setJointMotorControlArray(
                self.car_id, self.motor_joints, p.VELOCITY_CONTROL,
                targetVelocities=[target_speed] * len(self.motor_joints),
                forces=[150] * len(self.motor_joints))

            # HUD scalar defaults (overwritten if tracking succeeds)
            F_att_hud = F_rep_hud = F_tot_hud = 0.0

            if self.p0 is not None and len(self.p0) >= MIN_FEATURES:

                # ── Step 1: Track features (pyramidal LK) ──────────────────
                p1, status = self.tracker.track(self.prev_gray, curr_gray,
                                                 self.p0)
                mask       = status.flatten() == 1
                good_new   = p1[mask].reshape(-1, 2)
                good_old   = self.p0[mask].reshape(-1, 2)

                if len(good_new) >= MIN_FEATURES:

                    # ── Step 2: FOE (Sec. III-A) ───────────────────────────
                    foe = self._compute_foe(good_old, good_new)

                    foe_vis = (int(np.clip(foe[0], 0, self.W - 1)),
                               int(np.clip(foe[1], 0, self.H - 1)))
                    cv2.circle(rgb, foe_vis, 8, CLR_FOE, -1)
                    cv2.putText(rgb, "FOE",
                                (foe_vis[0] + 10, foe_vis[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                CLR_FOE, 1, cv2.LINE_AA)

                    # ── Step 3: Per-point speed ────────────────────────────
                    vx     = good_new[:, 0] - good_old[:, 0]
                    vy     = good_new[:, 1] - good_old[:, 1]
                    speeds = np.sqrt(vx**2 + vy**2) + 1e-5

                    # ── Step 4: Obstacle repulsive field (Sec. IV-B) ───────
                    F_obs_X, F_obs_Y = self._obstacle_forces(
                        good_old, good_new, speeds, foe, rgb)

                    # ── Step 5: Target / Attractive field (Sec. IV-A) ──────
                    pos, orn   = p.getBasePositionAndOrientation(self.car_id)
                    yaw        = p.getEulerFromQuaternion(orn)[2]
                    goal       = np.array([31.66, 0.0])

                    dist_goal  = (np.linalg.norm(goal - np.array(pos[:2]))
                                  + 1e-5)
                    theta_goal = np.arctan2(goal[1] - pos[1],
                                            goal[0] - pos[0])

                    F_att    = ALPHA_ATT * dist_goal          # Eq. 8
                    F_att_X  = F_att * np.cos(theta_goal - yaw)
                    F_att_Y  = F_att * np.sin(theta_goal - yaw)

                    # ── Step 6: Road boundary field (Sec. IV-C) ────────────
                    F_road_Y = self._road_force_y(float(pos[1]))

                    # ── Step 7: Total force (Eq. 19-20) ───────────────────
                    F_total_X = F_att_X + LAMBDA_X * F_obs_X
                    F_total_Y = F_att_Y + LAMBDA_Y * F_obs_Y + F_road_Y

                    # ── Step 8: GTSMC → steering (Eq. 27-30) ──────────────
                    steer, psi_d = self._gtsmc_steer(
                        F_total_X, F_total_Y, yaw)

                    p.setJointMotorControlArray(
                        self.car_id, self.steer_joints, p.POSITION_CONTROL,
                        targetPositions=[steer] * len(self.steer_joints))

                    # HUD values (lateral components are most informative)
                    F_att_hud = float(F_att_Y)
                    F_rep_hud = float(F_obs_Y + F_road_Y)
                    F_tot_hud = float(F_total_Y)

                    # Status text overlay
                    cv2.putText(
                        rgb,
                        (f"steer={np.degrees(steer):+.1f}deg  "
                         f"psi_d={np.degrees(psi_d):+.1f}deg  "
                         f"y={pos[1]:+.2f}m  x={pos[0]:.1f}m"),
                        (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (220, 220, 220), 1, cv2.LINE_AA)

                # ── Feature management: re-detect when needed ───────────────
                if (self.frame_idx % REDETECT_EVERY == 0
                        or len(good_new) < MIN_FEATURES):
                    self.p0 = self._detect(curr_gray)
                else:
                    self.p0 = good_new.reshape(-1, 1, 2).astype(np.float32)

            else:
                # Too few features — fall back to fresh detection
                self.p0 = self._detect(curr_gray)

            # ── Render HUD and display ──────────────────────────────────────
            _draw_force_panel(rgb, F_att_hud, F_rep_hud, F_tot_hud,
                              self.H, self.W)

            cv2.imshow("Flow Controller",
                       cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.prev_gray = curr_gray.copy()
            p.stepSimulation()
            time.sleep(self._dt)

        cv2.destroyAllWindows()
        print("[FlowController] Finished.")


# ──────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    car_id, steer_j, motor_j = setup_simulation(gui=True)
    ctrl = FlowController(car_id, steer_j, motor_j)
    ctrl.navigate(target_speed=10.0)