"""
flow_controller.py
──────────────────────────────────────────────────────────────────────────────
Visual Potential Field controller for autonomous driving in PyBullet.

Based on:
  Capito, Ozguner, Redmill — "Optical Flow based Visual Potential Field
  for Autonomous Driving" (IV 2020)

  Sazbon, Rotstein, Rivlin — "Finding the Focus of Expansion and
  Estimating Range using Optical Flow" (MVA 2004)

Feature strategy : Fixed uniform pixel grid  (stable, texture-independent).
Obstacle detect  : Flow mag → Otsu threshold → Gaussian smooth → Sobel
                   gradient  g(x,y,t) = ∇(G*O)  (Eq. 4-6).
Controller       : Gradient Tracking Sliding Mode (GTSMC), Sec. VI.

Per-frame pipeline
──────────────────
  1.  Render forward camera; convert to grayscale.
  2.  LK-track the fixed pixel grid from prev → curr frame.
  3.  Compute FOE via least-squares on flow vectors  (Sec. III-A).
  4.  Scatter per-point speeds onto a dense HxW canvas; Gaussian fill.
  5.  Otsu threshold → binary obstacle map  O(x,y,t)         (Sec. III-C).
  6.  Gaussian blur O (σ = W/2) then Sobel → g(x,y,t)       (Eq. 4-6).
  7.  Compute TTC per tracked grid point                     (Eq. 3).
  8.  Build visual potential fields:
        a. Attractive   U_att = ½α·r²  →  F_att = α·dist (Eq. 7-8)
        b. Repulsive    F_rep = γ·Σg / (|R|·ΣTTC)         (Eq. 9)
        c. Road Morse   F_road = −∇(U_sr + U_sl)           (Eq. 10-11, 18)
  9.  Total force (Eq. 19-20):
            F_T = F_att − F_rep − λ·F_road
  10. Transform F_T to global frame  (Eq. 21).
  11. Desired heading ψd = atan2(FY′, FX′)                  (Eq. 27).
  12. GTSMC sliding manifold → steering angle                (Eq. 28-30).

Key fixes in this version
─────────────────────────
  • Grid tracking instead of Shi-Tomasi: uniform pixel grid gives homogeneous
    coverage without depending on scene texture.  Grid refreshes every
    GRID_REFRESH_N frames so drift does not accumulate.

  • Corrected F_att: quadratic potential U = ½α·r² gives F_att = α·dist
    in the direction toward the goal, decomposed into the vehicle LOCAL frame
    as F_att_X = α·dist·cos(θ_goal − ψ), F_att_Y = α·dist·sin(θ_goal − ψ).

  • Corrected F_rep sign: g = ∇(G*O) points TOWARD the obstacle centre
    (positive gradient = uphill of Gaussian blob).  Eq. 9 is SUBTRACTED from
    F_att (Eq. 19-20), so the net effect is repulsion AWAY from the obstacle.
    Image +x maps to world −Y (rightward in image = rightward for driver =
    world −Y when +Y is left), so F_rep_world_Y = −γ·Σg_x / (|R|·ΣTTC).

  • Corrected road Morse force: simple exponential walls that are guaranteed
    to have the right sign — right boundary pushes toward +Y (left), left
    boundary pushes toward −Y (right).

  • Corrected GTSMC (Eq. 21 + 27): forces are transformed from LOCAL to
    GLOBAL frame before computing ψd.  Previously psi_d was computed in the
    local frame and compared directly with the global yaw — a frame mismatch.

  • Corrected total force (Eq. 19-20): repulsive forces are SUBTRACTED, not
    added.  Previously the code added F_rep, which doubled the attraction
    toward obstacles rather than repelling them.
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
CLR_FOE      = (255,   0,   0)   # blue dot   – FOE marker
CLR_FLOW_BG  = (0,   180,   0)   # dim-green  – background flow trail
CLR_FLOW_OBS = (0,     0, 220)   # red        – obstacle   flow trail
CLR_OBS_RING = (0,     0, 255)   # bright-red – obstacle ring


# ── Grid parameters ─────────────────────────────────────────────────────────
GRID_STEP      = 15   # pixels between adjacent grid nodes
GRID_MARGIN    = 10   # pixel margin kept clear at image border
GRID_REFRESH_N = 20   # full grid reset every N frames
MIN_GRID_PTS   = 15   # if fewer points survive LK, reset to full grid


# ── Obstacle-detection parameters ───────────────────────────────────────────
# O(x,y,t) is segmented via Otsu on the dense flow-magnitude image;
# no hand-tuned speed threshold is required.
TTC_THRESHOLD = 20.0   # ignore obstacle grid points with TTC > this (frames)


# ── Road potential field parameters ─────────────────────────────────────────
#   PyBullet road half-extents = 1.16 m; car spawns at y = 0.
#   ROAD_HALF_WIDTH is set slightly inside the hard edge for a safety margin.
ROAD_HALF_WIDTH = 0.55   # |y| at which the Morse wall begins to fire (m)
A_MORSE         = 0.5    # Morse potential depth A      (Table I, Capito 2020)
B_MORSE         = 8.0    # Morse variance parameter b   (tuned for sim scale)


# ── Force weights / gains ───────────────────────────────────────────────────
ALPHA_ATT  = 1.0     # attractive force scaling  α          (Eq. 8)
GAMMA_OBS  = 200.0   # repulsive obstacle gain   γ          (Eq. 9)
                      # (scaled for gradient values normalised to [0,1])
LAMBDA_X   = 1.0     # longitudinal force weight λX         (Eq. 19)
LAMBDA_Y   = 1.0     # lateral     force weight  λY         (Eq. 20)


# ── GTSMC parameters ────────────────────────────────────────────────────────
CR          = 0.8   # rotational-manifold constant cr   (Eq. 28)
U0          = 0.8   # control amplitude u0              (Eq. 30)
STEER_LIMIT = 0.6   # hard clamp on steering angle (rad)


# ═══════════════════════════════════════════════════════════════════════════
#  HUD helpers
# ═══════════════════════════════════════════════════════════════════════════

def _draw_force_arrow(img, origin, force_val, scale=60.0,
                      color=(255, 255, 255), label="",
                      thickness=2, tip_length=0.35):
    ox, oy = int(origin[0]), int(origin[1])
    ex     = int(ox + force_val * scale)
    cv2.arrowedLine(img, (ox, oy), (ex, oy), color,
                    thickness, tipLength=tip_length)
    if label:
        cv2.putText(img, f"{label}: {force_val:+.2f}",
                    (ox + 4, oy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def _draw_force_panel(rgb, f_att, f_rep, f_total, H, W):
    """Semi-transparent bottom panel showing the three lateral force arrows."""
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

    for ry, fv, lbl, clr in zip(rows, forces, labels, colors):
        _draw_force_arrow(rgb, (cx, ry), fv,
                          scale=60.0, color=clr, label=lbl)


# ═══════════════════════════════════════════════════════════════════════════
#  FlowController
# ═══════════════════════════════════════════════════════════════════════════

class FlowController:
    """
    Optical-flow Visual Potential Field controller — Capito et al. (IV 2020).

    Tracking
    --------
    A fixed uniform pixel grid is LK-tracked each frame.  The grid is reset
    every GRID_REFRESH_N frames so accumulated drift stays bounded.  This
    avoids dependence on scene texture (Shi-Tomasi failures on flat roads)
    while maintaining uniform spatial coverage for the Otsu obstacle map.

    Obstacle field  (Eq. 4-6, 9)
    -----------------------------
    Per-point flow speeds are scattered onto a dense canvas and Gaussian-
    filled to build a smooth magnitude image.  Otsu thresholding gives the
    binary obstacle map O(x,y,t).  A wide Gaussian (σ = W/2) is convolved
    with O, and its Sobel gradient g = ∇(G*O) gives the spatial direction
    and magnitude of the obstacle influence at every pixel.  g points TOWARD
    the obstacle centre; Eq. 9 defines F_rep from this gradient, and it is
    SUBTRACTED in Eq. 19-20 to produce the repulsive net force.

    Coordinate convention
    ---------------------
    Vehicle local frame: X = forward, Y = left (+Y = left of driver).
    Image frame: i = rightward, j = downward.
    When the camera faces forward: image +i  ↔  world −Y (right in image
    = right of driver = world −Y).
    Therefore  F_rep_world_Y = −(image-x component of γ·Σg / |R|·ΣTTC).

    GTSMC  (Eq. 21, 27-30)
    -----------------------
    Local-frame total forces are first rotated to the GLOBAL frame (Eq. 21)
    before computing the desired heading ψd = atan2(FY′, FX′).  The
    rotational sliding manifold sr = cr·ψe + ψ̇d is driven to zero by a
    tanh-smoothed control law to suppress chattering.
    """

    def __init__(self, car_id, steer_joints, motor_joints):
        self.car_id       = car_id
        self.steer_joints = steer_joints
        self.motor_joints = motor_joints

        # LK tracker — window 25 px, 3 pyramid levels (Capito Table II)
        self.tracker = LucasKanadeTracker_1(win=25, levels=3,
                                             max_iter=20, eps=0.03)

        self.W, self.H = 320, 240      # camera resolution (pixels)
        self.frame_idx = 0
        self.prev_gray = None

        # Build fixed grid and initialise p0
        self.grid_home = self._build_grid()               # (N,2) float32
        self.p0 = self.grid_home.reshape(-1, 1, 2).astype(np.float32)

        # GTSMC integration state
        self._psi_d_prev  = 0.0        # previous desired heading  (for ψ̇d)
        self._steer_angle = 0.0        # integrated steering angle (Eq. 29)
        self._dt          = 1.0 / 60.0 # simulation timestep (s)

    # ── Grid ───────────────────────────────────────────────────────────────

    def _build_grid(self):
        """
        Return an (N,2) float32 array of (x,y) pixel coordinates spaced
        GRID_STEP apart, leaving GRID_MARGIN pixels of border on each side.
        """
        xs = np.arange(GRID_MARGIN, self.W - GRID_MARGIN, GRID_STEP)
        ys = np.arange(GRID_MARGIN, self.H - GRID_MARGIN, GRID_STEP)
        xv, yv = np.meshgrid(xs, ys)
        return np.column_stack((xv.ravel(), yv.ravel())).astype(np.float32)

    # ── Camera ─────────────────────────────────────────────────────────────

    def _get_frame(self):
        """
        Render the forward camera rigidly attached to the car.

        The camera is offset 0.4 m forward and 0.35 m upward from the car
        body origin in the car's local frame, looking 2.0 m ahead at the
        same height.

        Returns
        -------
        gray : HxW  uint8
        rgb  : HxWx3 uint8  (RGB, for visualisation)
        """
        pos, orn   = p.getBasePositionAndOrientation(self.car_id)
        R          = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        cam_pos    = np.array(pos) + R @ [0.4, 0.0, 0.35]
        target_pos = np.array(pos) + R @ [2.0, 0.0, 0.35]

        view = p.computeViewMatrix(
            cam_pos.tolist(), target_pos.tolist(), [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(
            fov=90, aspect=self.W / self.H, nearVal=0.1, farVal=100)

        raw   = p.getCameraImage(self.W, self.H, view, proj,
                                 renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb   = raw[2][:, :, :3]
        gray  = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return gray, rgb.copy()

    # ── FOE — Sec. III-A ───────────────────────────────────────────────────

    def _compute_foe(self, p0, p1):
        """
        Least-squares Focus of Expansion from tracked correspondences.

        For each point i with position (x,y) and flow (vx,vy):
            a_i0 = vy,  a_i1 = −vx,  b_i = x·vy − y·vx
        FOE = (AᵀA)⁻¹ Aᵀb

        Falls back to the image centre when the system is under-determined.
        """
        v      = p1 - p0
        x, y   = p0[:, 0], p0[:, 1]
        vx, vy = v[:, 0],  v[:, 1]

        A   = np.column_stack((vy, -vx))        # (N,2)
        b   = x * vy - y * vx                   # (N,)
        ATA = A.T @ A                            # (2,2)

        det = ATA[0, 0] * ATA[1, 1] - ATA[0, 1] ** 2
        if abs(det) < 1e-6:
            return np.array([self.W / 2.0, self.H / 2.0])
        try:
            foe = np.linalg.solve(ATA, A.T @ b)
            foe[0] = np.clip(foe[0], -self.W,     2 * self.W)
            foe[1] = np.clip(foe[1], -self.H,     2 * self.H)
            return foe
        except np.linalg.LinAlgError:
            return np.array([self.W / 2.0, self.H / 2.0])

    # ── TTC — Eq. 3 ────────────────────────────────────────────────────────

    @staticmethod
    def _ttc(x, y, vx, vy, foe):
        """
        Per-point time-to-contact (Eq. 3):
            TTC_i = ‖p_i − FOE‖ / ‖v_i‖
        """
        dist = np.sqrt((x - foe[0]) ** 2 + (y - foe[1]) ** 2) + 1e-5
        spd  = np.sqrt(vx ** 2 + vy ** 2) + 1e-5
        return dist / spd

    # ── Obstacle detection pipeline — Eq. 4-6 ─────────────────────────────

    def _build_flow_magnitude_image(self, good_old, speeds):
        """
        Scatter sparse per-point flow speeds onto a dense HxW float32 image,
        then apply a small Gaussian fill to bridge the gaps between grid points.

        Parameters
        ----------
        good_old : (N,2) float32 — previous grid positions
        speeds   : (N,)  float32 — per-point flow magnitudes

        Returns
        -------
        mag_img : HxW float32
        """
        mag_img = np.zeros((self.H, self.W), dtype=np.float32)
        xi = np.clip(good_old[:, 0].astype(int), 0, self.W - 1)
        yi = np.clip(good_old[:, 1].astype(int), 0, self.H - 1)
        mag_img[yi, xi] = speeds
        # Light Gaussian fill to smooth gaps between grid points
        mag_img = cv2.GaussianBlur(mag_img, (7, 7), 0)
        return mag_img

    def _build_obstacle_map(self, mag_img):
        """
        Binary obstacle map O(x,y,t) via Otsu thresholding (Sec. III-C).

        Otsu maximises the inter-class variance between background flow
        (low speed) and obstacle flow (high speed).  No hand-tuned threshold
        is needed.

        Parameters
        ----------
        mag_img : HxW float32 — dense flow magnitude image

        Returns
        -------
        O : HxW uint8 — 255 at obstacle pixels, 0 elsewhere
        """
        norm = cv2.normalize(mag_img, None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)
        _, O = cv2.threshold(norm, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return O

    def _gaussian_gradient(self, O):
        """
        Compute g(x,y,t) = ∇(G*O)  (Eq. 6).

        σ = W/2 as specified in the paper.  Sobel gives ∂(G*O)/∂x and
        ∂(G*O)/∂y.  Both components are normalised to [−1, 1] so that
        GAMMA_OBS is independent of image brightness scale.

        Note: g points TOWARD the obstacle centre (positive gradient
        = uphill direction of the Gaussian blob).  Negation is NOT applied
        here — the sign is handled correctly by subtracting F_rep in the
        total force formula (Eq. 19-20).

        Returns
        -------
        gx, gy : HxW float32 — normalised Sobel gradient components of G*O
        """
        sigma   = self.W / 2.0
        blurred = cv2.GaussianBlur(O.astype(np.float32), (0, 0), sigma)
        gx      = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy      = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)

        # Normalise to [−1, 1] for consistent GAMMA_OBS scaling
        g_max   = max(float(np.abs(gx).max()), float(np.abs(gy).max()), 1e-5)
        return gx / g_max, gy / g_max

    # ── Obstacle repulsive force — Sec. IV-B, Eq. 9 ───────────────────────

    def _obstacle_forces(self, good_old, good_new, speeds, foe, rgb=None):
        """
        Repulsive force from the Otsu obstacle field (Eq. 4-6 → Eq. 9).

        Pipeline
        --------
        1. Dense flow magnitude image from grid speeds.
        2. Otsu threshold → binary O(x,y,t).
        3. Gaussian (σ=W/2) + Sobel → g(x,y,t) = ∇(G*O)  (normalised).
        4. TTC per tracked grid point  (Eq. 3).
        5. Obstacle set A: grid points inside O AND TTC < TTC_THRESHOLD.
        6. Eq. 9:
               F_rep = γ · (1/|R|) · Σ_{i∈A} g(xi, yi) / Σ_{i∈A} TTC_i

        Image → world coordinate mapping
        ---------------------------------
        Camera faces forward.  Image +i (rightward) = vehicle −Y direction
        (right of driver = world −Y where +Y is left).

          F_rep_world_Y = −γ · Σg_x / (|R| · ΣTTC)

        g points TOWARD the obstacle, so after subtraction in Eq. 19-20 the
        net force pushes AWAY from the obstacle:
          − F_rep_world_Y  →  if obstacle is left, sum_gx < 0, F_rep_Y > 0,
          total_Y = F_att_Y − F_rep_Y decreases → steer right ✓

        No explicit longitudinal component is computed from gy: the TTC
        weighting already creates urgency (close obstacles → small TTC →
        large F_rep → large lateral correction → car steers away faster).

        Parameters
        ----------
        good_old, good_new : (N,2) float32
        speeds             : (N,)  float32
        foe                : (2,)  float64
        rgb                : optional HxWx3 for debug overlay

        Returns
        -------
        F_rep_X : float — longitudinal (0 here; TTC urgency handled laterally)
        F_rep_Y : float — lateral, in vehicle local frame
        """
        # ── Steps 1-3: obstacle map and gradient field ─────────────────
        mag_img        = self._build_flow_magnitude_image(good_old, speeds)
        O              = self._build_obstacle_map(mag_img)
        gx_img, gy_img = self._gaussian_gradient(O)   # normalised, NOT negated

        # ── Step 4: TTC per grid point ──────────────────────────────────
        x, y   = good_old[:, 0], good_old[:, 1]
        vx     = good_new[:, 0] - good_old[:, 0]
        vy     = good_new[:, 1] - good_old[:, 1]
        ttc    = self._ttc(x, y, vx, vy, foe)

        xi = np.clip(x.astype(int), 0, self.W - 1)
        yi_px = np.clip(y.astype(int), 0, self.H - 1)

        # ── Step 5: obstacle set A ──────────────────────────────────────
        in_obs  = (O[yi_px, xi] > 0) & (ttc < TTC_THRESHOLD)
        obs_idx = np.where(in_obs)[0]

        # ── Debug overlay ───────────────────────────────────────────────
        if rgb is not None:
            # Subtle red tint on obstacle pixels
            obs_pixels = O > 0
            rgb[obs_pixels] = np.minimum(
                rgb[obs_pixels].astype(np.int16) + [40, 0, 0], 255
            ).astype(np.uint8)
            for i in range(len(good_new)):
                nx, ny   = int(good_new[i, 0]), int(good_new[i, 1])
                ox_p     = int(good_old[i, 0])
                oy_p     = int(good_old[i, 1])
                clr      = CLR_FLOW_OBS if in_obs[i] else CLR_FLOW_BG
                cv2.line(rgb, (nx, ny), (ox_p, oy_p), clr, 1)
                cv2.circle(rgb, (nx, ny), 2, clr, -1)
            for i in obs_idx:
                cv2.circle(rgb, (int(x[i]), int(y[i])), 6, CLR_OBS_RING, 2)

        if len(obs_idx) == 0:
            return 0.0, 0.0

        # ── Step 6: Eq. 9 ───────────────────────────────────────────────
        region  = float(len(obs_idx))
        sum_gx  = float(np.sum(gx_img[yi_px[obs_idx], xi[obs_idx]]))
        sum_ttc = float(np.sum(ttc[obs_idx])) + 1e-5

        # g_x points toward obstacle center in image space.
        # Image +x = world −Y  →  negate for world-Y component.
        # This value is then SUBTRACTED in Eq. 19-20 to give repulsion.
        F_rep_Y = -GAMMA_OBS * sum_gx / (region * sum_ttc)
        F_rep_X = 0.0   # longitudinal deceleration left to TTC-driven lateral urgency

        return F_rep_X, F_rep_Y

    # ── Road potential field — Sec. IV-C, Eq. 10-11, 18 ───────────────────

    def _road_force_y(self, y_car):
        """
        Lateral repulsive force from the road boundaries.

        Uses simple exponential walls derived from the modified Morse
        potential (Eq. 11, 18).  Sign is explicit so the direction is
        always correct regardless of which side the car is on:

            F_from_right  = +A · exp(−b · dist_from_right_wall)  → pushes +Y (left)
            F_from_left   = −A · exp(−b · dist_from_left_wall)   → pushes −Y (right)

        At y=0 (centre): both terms are equal, net = 0.
        Near right wall (y → −ROAD_HALF_WIDTH): F_from_right dominates → push left.
        Near left  wall (y → +ROAD_HALF_WIDTH): F_from_left dominates  → push right.

        This is returned as a world-Y force and SUBTRACTED in Eq. 20.
        """
        yr = -ROAD_HALF_WIDTH   # right wall (−Y side)
        yl =  ROAD_HALF_WIDTH   # left  wall (+Y side)

        dist_right = max(float(y_car - yr), 1e-3)   # distance inside right wall
        dist_left  = max(float(yl - y_car), 1e-3)   # distance inside left  wall

        F_from_right =  A_MORSE * np.exp(-B_MORSE * dist_right)  # pushes +Y (left)
        F_from_left  = -A_MORSE * np.exp(-B_MORSE * dist_left)   # pushes −Y (right)

        return F_from_right + F_from_left

    # ── Attractive force — Sec. IV-A, Eq. 7-8 ─────────────────────────────

    @staticmethod
    def _attractive_force(pos, yaw, goal):
        """
        Attractive force toward the goal in the vehicle LOCAL frame.

        Quadratic potential  U_att = ½α·r²  →  ‖F_att‖ = α·r  (Eq. 7-8).
        Direction is toward the goal; decomposed into local (X=forward, Y=left):

            θ_goal  = atan2(goal_y − car_y,  goal_x − car_x)  [global frame]
            F_att_X = α · dist · cos(θ_goal − ψ)              [local forward]
            F_att_Y = α · dist · sin(θ_goal − ψ)              [local lateral]

        Parameters
        ----------
        pos  : (3,) position of car in world frame
        yaw  : float, car heading (rad)
        goal : (2,) [goal_x, goal_y] in world frame

        Returns
        -------
        F_att_X, F_att_Y : float — attractive force in LOCAL vehicle frame
        """
        dx        = goal[0] - pos[0]
        dy        = goal[1] - pos[1]
        dist      = np.sqrt(dx ** 2 + dy ** 2) + 1e-5
        theta_g   = np.arctan2(dy, dx)              # global angle toward goal

        F_mag     = ALPHA_ATT * dist                 # α · r  (Eq. 8)
        F_att_X   = F_mag * np.cos(theta_g - yaw)   # local X (forward)
        F_att_Y   = F_mag * np.sin(theta_g - yaw)   # local Y (lateral)
        return F_att_X, F_att_Y

    # ── GTSMC — Sec. VI, Eq. 21 + 27-30 ───────────────────────────────────

    def _gtsmc_steer(self, F_total_X, F_total_Y, yaw):
        """
        Gradient Tracking Sliding Mode Controller.

        Step A — Local → Global frame transformation  (Eq. 21):
            FX′ = F_T_X · cos ψ + F_T_Y · sin ψ
            FY′ = −F_T_X · sin ψ + F_T_Y · cos ψ

        Step B — Desired heading in global frame  (Eq. 27):
            ψd = atan2(FY′, FX′)

        Step C — Error and sliding manifold  (Eq. 28):
            ψe = ψ − ψd      (wrapped to [−π, π])
            sr = cr · ψe + ψ̇d

        Step D — Control law  (Eq. 29-30, with tanh soft-sign):
            u = −u0 · tanh(5 · sr)
            δ̇ = u   →   δ = ∫u dt   (clamped to ±STEER_LIMIT)

        Parameters
        ----------
        F_total_X, F_total_Y : float — total force in LOCAL vehicle frame
        yaw                  : float — current vehicle heading (rad, global)

        Returns
        -------
        steer_angle : float — target steering angle (rad) sent to PyBullet
        psi_d       : float — desired global heading (rad, for HUD)
        """
        # Step A: rotate local forces to global frame (Eq. 21)
        FX_prime = F_total_X * np.cos(yaw) + F_total_Y * np.sin(yaw)
        FY_prime = -F_total_X * np.sin(yaw) + F_total_Y * np.cos(yaw)

        # Step B: desired heading in GLOBAL frame (Eq. 27)
        psi_d = np.arctan2(FY_prime, max(abs(FX_prime), 0.1))

        # Step C: heading error (global frame) and sliding manifold (Eq. 28)
        psi_e = yaw - psi_d
        psi_e = (psi_e + np.pi) % (2 * np.pi) - np.pi   # wrap to [−π, π]

        psi_d_dot        = (psi_d - self._psi_d_prev) / self._dt
        self._psi_d_prev = psi_d

        sr = CR * psi_e + psi_d_dot

        # Step D: tanh-smoothed control (Eq. 29-30)
        u = -U0 * np.tanh(5.0 * sr)

        self._steer_angle += u * self._dt
        self._steer_angle  = np.clip(self._steer_angle,
                                     -STEER_LIMIT, STEER_LIMIT)
        return self._steer_angle, psi_d

    # ── Main navigation loop ───────────────────────────────────────────────

    def navigate(self, target_speed: float = 10.0):
        """
        Drive from the spawn position to the end wall (x ≈ 31.7 m) while
        weaving around the five slalom obstacles using visual potential fields.

        Press Q in the OpenCV window to stop early.
        """
        print("[FlowController] Starting — press Q to quit")
        self.prev_gray, _ = self._get_frame()
        # Initialise p0 to the full grid
        self.p0 = self.grid_home.reshape(-1, 1, 2).astype(np.float32)

        while True:
            self.frame_idx += 1
            curr_gray, rgb = self._get_frame()

            # ── Constant forward drive ──────────────────────────────────
            p.setJointMotorControlArray(
                self.car_id, self.motor_joints, p.VELOCITY_CONTROL,
                targetVelocities=[target_speed] * len(self.motor_joints),
                forces=[150] * len(self.motor_joints))

            F_att_hud = F_rep_hud = F_tot_hud = 0.0

            if self.p0 is not None and len(self.p0) >= MIN_GRID_PTS:

                # ── Step 2: LK-track the grid ───────────────────────────
                p1, status = self.tracker.track(
                    self.prev_gray, curr_gray, self.p0)
                ok       = status.flatten() == 1
                good_new = p1[ok].reshape(-1, 2)
                good_old = self.p0[ok].reshape(-1, 2)

                if len(good_new) >= MIN_GRID_PTS:

                    # ── Step 3: FOE ────────────────────────────────────
                    foe = self._compute_foe(good_old, good_new)
                    foe_vis = (int(np.clip(foe[0], 0, self.W - 1)),
                               int(np.clip(foe[1], 0, self.H - 1)))
                    cv2.circle(rgb, foe_vis, 8, CLR_FOE, -1)
                    cv2.putText(rgb, "FOE",
                                (foe_vis[0] + 10, foe_vis[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                CLR_FOE, 1, cv2.LINE_AA)

                    # Per-point flow speeds for obstacle pipeline
                    vx     = good_new[:, 0] - good_old[:, 0]
                    vy     = good_new[:, 1] - good_old[:, 1]
                    speeds = np.sqrt(vx ** 2 + vy ** 2) + 1e-5

                    # ── Steps 4-6: Otsu + Gaussian → F_rep  (Eq. 4-6, 9) ──
                    F_rep_X, F_rep_Y = self._obstacle_forces(
                        good_old, good_new, speeds, foe, rgb)

                    # ── Step 8a: Attractive force  (Eq. 7-8) ───────────
                    pos, orn = p.getBasePositionAndOrientation(self.car_id)
                    yaw      = p.getEulerFromQuaternion(orn)[2]
                    goal     = np.array([31.66, 0.0])

                    F_att_X, F_att_Y = self._attractive_force(
                        pos, yaw, goal)

                    # ── Step 8c: Road Morse force  (Eq. 10-11, 18) ─────
                    F_road_Y = self._road_force_y(float(pos[1]))

                    # ── Step 9: Total force  (Eq. 19-20) ───────────────
                    # F_T = F_att − F_rep − λ·F_road
                    # Both F_rep and F_road are repulsive → SUBTRACTED.
                    F_total_X = F_att_X - LAMBDA_X * F_rep_X
                    F_total_Y = F_att_Y - LAMBDA_Y * F_rep_Y - F_road_Y

                    # ── Steps 10-12: GTSMC → steer  (Eq. 21, 27-30) ───
                    steer, psi_d = self._gtsmc_steer(
                        F_total_X, F_total_Y, yaw)

                    p.setJointMotorControlArray(
                        self.car_id, self.steer_joints, p.POSITION_CONTROL,
                        targetPositions=[steer] * len(self.steer_joints))

                    # HUD — show lateral components for clarity
                    F_att_hud = float(F_att_Y)
                    F_rep_hud = float(F_rep_Y + F_road_Y)
                    F_tot_hud = float(F_total_Y)

                    cv2.putText(
                        rgb,
                        (f"steer={np.degrees(steer):+.1f}d  "
                         f"psi_d={np.degrees(psi_d):+.1f}d  "
                         f"y={pos[1]:+.2f}m  x={pos[0]:.1f}m"),
                        (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (220, 220, 220), 1, cv2.LINE_AA)

                # ── Grid management ─────────────────────────────────────
                # Reset to full grid periodically or when too many points lost
                if (self.frame_idx % GRID_REFRESH_N == 0
                        or len(good_new) < MIN_GRID_PTS):
                    self.p0 = self.grid_home.reshape(-1, 1, 2).astype(
                        np.float32)
                else:
                    self.p0 = good_new.reshape(-1, 1, 2).astype(np.float32)

            else:
                # Full reset
                self.p0 = self.grid_home.reshape(-1, 1, 2).astype(np.float32)

            # ── HUD and display ─────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    car_id, steer_j, motor_j = setup_simulation(gui=True)
    ctrl = FlowController(car_id, steer_j, motor_j)
    ctrl.navigate(target_speed=10.0)