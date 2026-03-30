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
Obstacle detect  : Otsu segmentation → Gaussian gradient field (Eq. 4-6).
Controller       : Gradient Tracking Sliding Mode (GTSMC), Sec. VI.

Full pipeline per frame
───────────────────────
  1.  Capture grayscale image from forward-mounted camera.
  2.  Track features: Shi-Tomasi → pyramidal Lucas-Kanade (cv2 back-end).
  3.  Compute FOE via least-squares (Sec. III-A).
  4.  Scatter sparse flow speeds → dense magnitude image.
  5.  Otsu threshold → binary obstacle map  O(x,y,t)        (Sec. III-C).
  6.  Gaussian smooth O (σ = W/2), then Sobel → g(x,y,t)   (Eq. 4-6).
  7.  Compute TTC at each tracked point                     (Eq. 3).
  8.  Compute Visual Potential Field:
        a. Target / Attractive field   (Sec. IV-A, Eq. 7-8)
        b. Obstacle repulsive field    Σg / ΣTTC             (Eq. 9)
        c. Road boundary Morse field   (Sec. IV-C, Eq. 10-11, 18)
  9.  Sum forces → desired heading ψd  (Eq. 27).
  10. GTSMC: sliding manifold sr → steering angle           (Eq. 28-30).
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


# ── Obstacle-detection parameters ──────────────────────────────────────────
# O(x,y,t) is built via Otsu thresholding of the dense flow-magnitude image.
# No hand-tuned speed threshold is required (Sec. III-C).
TTC_THRESHOLD = 20.0   # ignore obstacle points with TTC > this (frames)


# ── Road potential field (Morse, Table I — Capito 2020) ────────────────────
#   Road half-extents in simulation = 1.16 m (create_road_and_obstacles).
#   Car starts at y = 0; slalom obstacles at y = ±0.38 m.
ROAD_HALF_WIDTH = 0.55   # |y| at which Morse potential begins to fire (m)
A_MORSE         = 0.5    # Morse depth  A   (Table I)
B_MORSE         = 8.0    # Morse variance b  (tuned for simulation scale)


# ── Force weights / gains ───────────────────────────────────────────────────
ALPHA_ATT  = 1.0       # attractive force scaling  α   (Eq. 8)
GAMMA_OBS  = 10000.0   # repulsive obstacle gain   γ   (Eq. 9)
LAMBDA_X   = 1.0       # longitudinal force weight λX  (Eq. 19)
LAMBDA_Y   = 1.0       # lateral     force weight  λY  (Eq. 20)


# ── GTSMC parameters ────────────────────────────────────────────────────────
CR          = 0.8   # rotational-manifold gain cr   (Eq. 28)
U0          = 0.8   # control amplitude u0          (Eq. 30)
STEER_LIMIT = 0.6   # hard clamp on steering angle (rad)


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
    Optical-flow Visual Potential Field controller for PyBullet racecar.

    Implements the full pipeline from Capito et al. (IV 2020):
      • Feature tracking  — Shi-Tomasi + pyramidal LK
      • FOE estimation    — least-squares (Sec. III-A)
      • Obstacle map      — Otsu threshold on dense flow magnitude
      • Obstacle gradient — Gaussian smooth then Sobel → g(x,y,t) (Eq. 4-6)
      • Repulsive force   — Σg / ΣTTC over obstacle region (Eq. 9)
      • Road Morse field  — modified Morse potential (Eq. 10-11, 18)
      • Controller        — GTSMC lateral steering (Eq. 27-30)
    """

    def __init__(self, car_id, steer_joints, motor_joints):
        self.car_id       = car_id
        self.steer_joints = steer_joints
        self.motor_joints = motor_joints

        # LK tracker — window 25 px, 3 pyramid levels (Capito Table II)
        self.tracker   = LucasKanadeTracker_1(win=25, levels=3,
                                               max_iter=20, eps=0.03)
        self.prev_gray = None
        self.p0        = None          # (N,1,2) float32 tracked feature set

        self.W, self.H = 320, 240      # camera resolution (pixels)
        self.frame_idx = 0

        # GTSMC state
        self._psi_d_prev  = 0.0        # previous desired heading for ψ̇d
        self._steer_angle = 0.0        # integrated steering angle (Eq. 29)
        self._dt          = 1.0 / 60.0 # simulation timestep (s)

    # ─────────────────────────────────────────────────────────────────────
    #  Camera
    # ─────────────────────────────────────────────────────────────────────

    def _get_frame(self):
        """
        Render a forward-facing camera rigidly attached to the car.

        Camera is offset 0.4 m forward and 0.35 m upward from the car
        origin in the car's local frame, looking 2.0 m ahead.

        Returns
        -------
        gray : HxW  uint8 — grayscale image
        rgb  : HxWx3 uint8 — RGB image (for visualisation)
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
        rgb  = img_arr[2][:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return gray, rgb.copy()

    # ─────────────────────────────────────────────────────────────────────
    #  Feature management
    # ─────────────────────────────────────────────────────────────────────

    def _detect(self, gray):
        """
        Shi-Tomasi corner detection.

        Returns (N,1,2) float32.  Returns an empty array rather than None
        so callers never need to guard for None.
        """
        pts = cv2.goodFeaturesToTrack(gray, mask=None, **FEATURE_PARAMS)
        if pts is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return pts.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────
    #  FOE — Sec. III-A (Capito 2020)
    # ─────────────────────────────────────────────────────────────────────

    def _compute_foe(self, p0, p1):
        """
        Least-squares Focus of Expansion from tracked correspondences.

        For each point i with position (x,y) and flow (vx,vy):
            a_i0 = vy,   a_i1 = -vx,   b_i = x·vy − y·vx
        FOE = (AᵀA)⁻¹ Aᵀb

        Returns image-centre if the system is under-constrained.
        """
        v      = p1 - p0
        x, y   = p0[:, 0], p0[:, 1]
        vx, vy = v[:, 0],  v[:, 1]

        A   = np.column_stack((vy, -vx))
        b   = x * vy - y * vx
        ATA = A.T @ A

        try:
            det = ATA[0, 0] * ATA[1, 1] - ATA[0, 1] * ATA[1, 0]
            if abs(det) < 1e-6:
                return np.array([self.W / 2.0, self.H / 2.0])
            foe    = np.linalg.solve(ATA, A.T @ b)
            foe[0] = np.clip(foe[0], -self.W,    2 * self.W)
            foe[1] = np.clip(foe[1], -self.H,    2 * self.H)
            return foe
        except np.linalg.LinAlgError:
            return np.array([self.W / 2.0, self.H / 2.0])

    # ─────────────────────────────────────────────────────────────────────
    #  TTC — Eq. 3 (Capito 2020)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ttc(x, y, vx, vy, foe):
        """
        Per-point time-to-contact (Eq. 3):
            TTC_i = ‖(x,y) − FOE‖ / ‖(vx,vy)‖
        """
        dist = np.sqrt((x - foe[0])**2 + (y - foe[1])**2) + 1e-5
        spd  = np.sqrt(vx**2 + vy**2) + 1e-5
        return dist / spd

    # ─────────────────────────────────────────────────────────────────────
    #  Obstacle detection — Sec. III-C, Eq. 4-6 (Capito 2020)
    # ─────────────────────────────────────────────────────────────────────

    def _build_flow_magnitude_image(self, good_old, speeds):
        """
        Scatter sparse per-point flow speeds onto a dense HxW float32 image.

        Each tracked point writes its speed to the nearest integer pixel.
        A small Gaussian fill (7×7) bridges gaps between sparse samples so
        the subsequent Otsu threshold sees a smooth magnitude field.

        Parameters
        ----------
        good_old : (N,2) float32 — tracked point positions
        speeds   : (N,)  float32 — per-point flow magnitude

        Returns
        -------
        mag_img : HxW float32
        """
        mag_img = np.zeros((self.H, self.W), dtype=np.float32)
        xi = np.clip(good_old[:, 0].astype(int), 0, self.W - 1)
        yi = np.clip(good_old[:, 1].astype(int), 0, self.H - 1)
        mag_img[yi, xi] = speeds
        mag_img = cv2.GaussianBlur(mag_img, (7, 7), 0)
        return mag_img

    def _build_obstacle_map(self, mag_img):
        """
        Binary obstacle map  O(x,y,t)  via Otsu thresholding (Sec. III-C).

        Otsu's method finds the threshold that maximises inter-class variance
        between background (low flow) and obstacle (high flow) pixels.  No
        hand-tuned speed constant is required.

        Parameters
        ----------
        mag_img : HxW float32 — dense flow magnitude image

        Returns
        -------
        O : HxW uint8 — 255 where obstacle flow is detected, 0 elsewhere
        """
        norm = cv2.normalize(mag_img, None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)
        _, O = cv2.threshold(norm, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return O

    def _gaussian_gradient(self, O):
        """
        Compute  g(x,y,t) = ∇(G * O)  (Eq. 4-6).

        Steps (following the paper exactly):
          1. Convolve O with Gaussian G (Eq. 5):
                G*O  using cv2.GaussianBlur with σ = W/2
          2. Take the spatial gradient of G*O (Eq. 6):
                gx = ∂(G*O)/∂x,   gy = ∂(G*O)/∂y   via Sobel

        The Gaussian has high values at the obstacle centroid and decays
        outward.  Its gradient therefore points *toward* the centroid.
        We negate so the force points *away* — i.e. repulsive.

        σ = W/2 is specified by the paper as "half the image width".

        Parameters
        ----------
        O : HxW uint8 — binary obstacle map

        Returns
        -------
        gx, gy : HxW float32 — repulsive gradient field (negated)
        """
        sigma   = self.W / 2.0
        blurred = cv2.GaussianBlur(O.astype(np.float32), (0, 0), sigma)
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        return -gx, -gy   # negate → repulsive (away from obstacle centroid)

    # ─────────────────────────────────────────────────────────────────────
    #  Obstacle Potential Field — Sec. IV-B, Eq. 9
    # ─────────────────────────────────────────────────────────────────────

    def _obstacle_forces(self, good_old, good_new, speeds, foe, rgb=None):
        """
        Repulsive force from obstacles using the full Otsu + Gaussian
        gradient pipeline (Sec. III-C, Eq. 4-6 → Sec. IV-B, Eq. 9).

        Pipeline
        --------
        1. _build_flow_magnitude_image  — sparse speeds → dense mag image
        2. _build_obstacle_map          — Otsu threshold → binary O(x,y,t)
        3. _gaussian_gradient           — Gaussian blur + Sobel → g(x,y,t)
        4. _ttc                         — TTC at every tracked point (Eq. 3)
        5. Select region A: obstacle pixels (O > 0) with TTC < threshold
        6. Eq. 9:
               F_rep = γ · (1/|R|) · Σ_{i∈A} g(xi,yi,t)
                                    / Σ_{i∈A} TTC_i

        Force direction is determined geometrically by where the obstacle
        blob sits in the image — no explicit left/right sign logic needed.

        Image → world coordinate mapping
        ---------------------------------
        gx (image +x = right)  →  F_obs_Y (world lateral)
        gy (image +y = down)   →  F_obs_X (world longitudinal, negated
                                   so a low-in-image obstacle → slow down)

        Parameters
        ----------
        good_old, good_new : (N,2) float32 — point positions prev / curr
        speeds             : (N,)  float32 — per-point flow magnitude
        foe                : (2,)  float64 — focus of expansion (px)
        rgb                : optional HxWx3 uint8 — debug overlay target

        Returns
        -------
        F_obs_X, F_obs_Y : float — obstacle repulsive force (local frame)
        """
        # ── Steps 1-3: build obstacle map and gradient field ───────────
        mag_img        = self._build_flow_magnitude_image(good_old, speeds)
        O              = self._build_obstacle_map(mag_img)
        gx_img, gy_img = self._gaussian_gradient(O)

        # ── Step 4: TTC at every tracked point ─────────────────────────
        x, y   = good_old[:, 0], good_old[:, 1]
        vx     = good_new[:, 0] - good_old[:, 0]
        vy     = good_new[:, 1] - good_old[:, 1]
        ttc    = self._ttc(x, y, vx, vy, foe)

        xi = np.clip(x.astype(int), 0, self.W - 1)
        yi = np.clip(y.astype(int), 0, self.H - 1)

        # ── Step 5: select region A (obstacle pixels + short TTC) ──────
        in_obstacle = (O[yi, xi] > 0) & (ttc < TTC_THRESHOLD)

        # ── Debug overlay ──────────────────────────────────────────────
        if rgb is not None:
            # Subtle red tint on pixels belonging to the obstacle map
            obs_mask = O > 0
            rgb[obs_mask] = np.minimum(
                rgb[obs_mask].astype(np.int16) + [40, 0, 0], 255
            ).astype(np.uint8)
            # Colour-coded flow trails
            for i in range(len(good_new)):
                nx, ny = int(good_new[i, 0]), int(good_new[i, 1])
                ox_p   = int(good_old[i, 0])
                oy_p   = int(good_old[i, 1])
                clr    = CLR_FLOW_OBS if in_obstacle[i] else CLR_FLOW_BG
                cv2.line(rgb, (nx, ny), (ox_p, oy_p), clr, 1)
                cv2.circle(rgb, (nx, ny), 2, clr, -1)
                if in_obstacle[i]:
                    cv2.circle(rgb, (int(x[i]), int(y[i])),
                               6, CLR_OBS_RING, 2)

        # ── Step 6: Eq. 9 ──────────────────────────────────────────────
        obs_idx = np.where(in_obstacle)[0]
        if len(obs_idx) == 0:
            return 0.0, 0.0

        sum_ttc = float(np.sum(ttc[obs_idx]))  + 1e-5
        sum_gx  = float(np.sum(gx_img[yi[obs_idx], xi[obs_idx]]))
        sum_gy  = float(np.sum(gy_img[yi[obs_idx], xi[obs_idx]]))
        region  = float(len(obs_idx))

        # gx → lateral (Y);   −|gy| → longitudinal slow-down (X)
        F_obs_Y = GAMMA_OBS * sum_gx  / (region * sum_ttc)
        F_obs_X = GAMMA_OBS * (-abs(sum_gy)) / (region * sum_ttc)

        return F_obs_X, F_obs_Y

    # ─────────────────────────────────────────────────────────────────────
    #  Road Potential Field — Sec. IV-C, Eq. 10-11, 18
    # ─────────────────────────────────────────────────────────────────────

    def _road_force_y(self, y_car):
        """
        Repulsive lateral force from road boundaries.

        Modified Morse potential (Eq. 10-11):
            Usr = A (1 − e^{−b(y−yr)})²    right boundary (yr < 0)
            Usl = A (1 − e^{ b(y−yl)})²    left  boundary (yl > 0)

        Negative gradient (Eq. 18):
            ∂Usr/∂y =  2Ab e^{−b(y−yr)} (1 − e^{−b(y−yr)})
            ∂Usl/∂y = −2Ab e^{ b(y−yl)} (1 − e^{ b(y−yl)})
            F_road_y = −(∂Usr/∂y + ∂Usl/∂y)

        Parameters
        ----------
        y_car : float — current lateral position of the car (m)

        Returns
        -------
        F_road_y : float — lateral repulsive force pushing car to centre
        """
        yr = -ROAD_HALF_WIDTH
        yl =  ROAD_HALF_WIDTH

        dUsr_dy = (2 * A_MORSE * B_MORSE
                   * np.exp(-B_MORSE * (y_car - yr))
                   * (1 - np.exp(-B_MORSE * (y_car - yr))))

        dUsl_dy = (-2 * A_MORSE * B_MORSE
                   * np.exp(B_MORSE * (y_car - yl))
                   * (1 - np.exp(B_MORSE * (y_car - yl))))

        return -(dUsr_dy + dUsl_dy)

    # ─────────────────────────────────────────────────────────────────────
    #  GTSMC — Sec. VI, Eq. 27-30
    # ─────────────────────────────────────────────────────────────────────

    def _gtsmc_steer(self, F_total_X, F_total_Y, yaw):
        """
        Gradient Tracking Sliding Mode Controller.

        Desired heading (Eq. 27):
            ψd = atan2(F_Y, F_X)

        Orientation error:
            ψe = ψ − ψd   (wrapped to [−π, π])

        Rotational sliding manifold (Eq. 28):
            sr = cr·ψe + ψ̇d   (ψ̇d approximated by finite difference)

        Control law (Eq. 29-30, with tanh to suppress chattering):
            u  = −u0 · tanh(5 · sr)
            δ̇ = u   →   δ = ∫u dt   (clamped to ±STEER_LIMIT)

        Parameters
        ----------
        F_total_X, F_total_Y : float — total force components (local frame)
        yaw                  : float — current vehicle yaw (rad)

        Returns
        -------
        steer_angle : float — target steering angle (rad) for PyBullet
        psi_d       : float — desired heading (rad, for diagnostics)
        """
        psi_d = np.arctan2(F_total_Y, max(abs(F_total_X), 0.1))

        psi_e = yaw - psi_d
        psi_e = (psi_e + np.pi) % (2 * np.pi) - np.pi   # wrap to [−π, π]

        psi_d_dot        = (psi_d - self._psi_d_prev) / self._dt
        self._psi_d_prev = psi_d

        sr = CR * psi_e + psi_d_dot

        u = -U0 * np.tanh(5.0 * sr)

        self._steer_angle += u * self._dt
        self._steer_angle  = np.clip(self._steer_angle,
                                     -STEER_LIMIT, STEER_LIMIT)
        return self._steer_angle, psi_d

    # ─────────────────────────────────────────────────────────────────────
    #  Main navigation loop
    # ─────────────────────────────────────────────────────────────────────

    def navigate(self, target_speed: float = 10.0):
        """
        Drive the car from spawn to the end wall (x ≈ 31.7 m), avoiding
        the five slalom obstacles using the visual potential field.

        Parameters
        ----------
        target_speed : float
            Wheel velocity target (rad/s).  Default 10.0 gives a
            comfortable approach speed for the slalom layout.

        Controls
        --------
        Q — quit the simulation window.
        """
        print("[FlowController] Starting — press Q to quit")
        self.prev_gray, _ = self._get_frame()
        self.p0           = self._detect(self.prev_gray)

        while True:
            self.frame_idx += 1
            curr_gray, rgb = self._get_frame()

            # ── Constant forward drive ─────────────────────────────────
            p.setJointMotorControlArray(
                self.car_id, self.motor_joints, p.VELOCITY_CONTROL,
                targetVelocities=[target_speed] * len(self.motor_joints),
                forces=[150] * len(self.motor_joints))

            F_att_hud = F_rep_hud = F_tot_hud = 0.0

            if self.p0 is not None and len(self.p0) >= MIN_FEATURES:

                # ── Step 2: Track with pyramidal LK ───────────────────
                p1, status = self.tracker.track(
                    self.prev_gray, curr_gray, self.p0)
                mask     = status.flatten() == 1
                good_new = p1[mask].reshape(-1, 2)
                good_old = self.p0[mask].reshape(-1, 2)

                if len(good_new) >= MIN_FEATURES:

                    # ── Step 3: FOE ───────────────────────────────────
                    foe = self._compute_foe(good_old, good_new)

                    foe_vis = (int(np.clip(foe[0], 0, self.W - 1)),
                               int(np.clip(foe[1], 0, self.H - 1)))
                    cv2.circle(rgb, foe_vis, 8, CLR_FOE, -1)
                    cv2.putText(rgb, "FOE",
                                (foe_vis[0] + 10, foe_vis[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                CLR_FOE, 1, cv2.LINE_AA)

                    # Per-point flow magnitudes (needed by steps 4-6)
                    vx     = good_new[:, 0] - good_old[:, 0]
                    vy     = good_new[:, 1] - good_old[:, 1]
                    speeds = np.sqrt(vx**2 + vy**2) + 1e-5

                    # ── Steps 4-7: Otsu obstacle field (Eq. 4-6, 9) ──
                    F_obs_X, F_obs_Y = self._obstacle_forces(
                        good_old, good_new, speeds, foe, rgb)

                    # ── Step 8a: Attractive field (Eq. 7-8) ───────────
                    pos, orn   = p.getBasePositionAndOrientation(self.car_id)
                    yaw        = p.getEulerFromQuaternion(orn)[2]
                    goal       = np.array([31.66, 0.0])

                    dist_goal  = np.linalg.norm(
                        goal - np.array(pos[:2])) + 1e-5
                    theta_goal = np.arctan2(
                        goal[1] - pos[1], goal[0] - pos[0])

                    F_att   = ALPHA_ATT * dist_goal
                    F_att_X = F_att * np.cos(theta_goal - yaw)
                    F_att_Y = F_att * np.sin(theta_goal - yaw)

                    # ── Step 8c: Road Morse field (Eq. 10-11, 18) ─────
                    F_road_Y = self._road_force_y(float(pos[1]))

                    # ── Step 9: Total force (Eq. 19-20) ───────────────
                    F_total_X = F_att_X + LAMBDA_X * F_obs_X
                    F_total_Y = F_att_Y + LAMBDA_Y * F_obs_Y + F_road_Y

                    # ── Step 10: GTSMC → steering (Eq. 27-30) ─────────
                    steer, psi_d = self._gtsmc_steer(
                        F_total_X, F_total_Y, yaw)

                    p.setJointMotorControlArray(
                        self.car_id, self.steer_joints, p.POSITION_CONTROL,
                        targetPositions=[steer] * len(self.steer_joints))

                    # HUD scalars (lateral components most informative)
                    F_att_hud = float(F_att_Y)
                    F_rep_hud = float(F_obs_Y + F_road_Y)
                    F_tot_hud = float(F_total_Y)

                    cv2.putText(
                        rgb,
                        (f"steer={np.degrees(steer):+.1f}d  "
                         f"psi_d={np.degrees(psi_d):+.1f}d  "
                         f"y={pos[1]:+.2f}m  x={pos[0]:.1f}m"),
                        (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (220, 220, 220), 1, cv2.LINE_AA)

                # ── Feature re-detection ───────────────────────────────
                if (self.frame_idx % REDETECT_EVERY == 0
                        or len(good_new) < MIN_FEATURES):
                    self.p0 = self._detect(curr_gray)
                else:
                    self.p0 = good_new.reshape(-1, 1, 2).astype(np.float32)

            else:
                self.p0 = self._detect(curr_gray)

            # ── HUD + display ──────────────────────────────────────────
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