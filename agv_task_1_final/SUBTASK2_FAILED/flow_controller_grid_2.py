"""
flow_controller_grid.py
────────────────────────────────────────────────────────────────────────────
Feature strategy : PERMANENT fixed-coordinate grid
  - Every grid node is a fixed (x, y) pixel coordinate burned in at startup.
  - After each LK step, any node that was lost or drifted is snapped
    EXACTLY back to its home pixel coordinate – not "nearby", exactly there.
  - The active point array is always the full grid; only the flow vectors
    differ between frames.

Obstacle detection: Flow-speed anomaly vs scene median.
"""

import cv2
import numpy as np
import pybullet as p
import time

from lucas_kanade import LucasKanadeTracker_1
from simulation_setup import setup_simulation


# ── Colour palette ────────────────────────────────────────────────────────────
CLR_ATT       = (0,   220,   0)   # green  – attractive force
CLR_REP       = (0,    60, 255)   # red    – repulsive force
CLR_TOT       = (0,   220, 255)   # yellow – total force
CLR_FOE       = (255,   0,   0)   # blue   – FOE
CLR_BG_DOT    = (40,  140,  40)   # dim green dot  – background node
CLR_OBS_DOT   = (0,     0, 255)   # red dot        – obstacle node
CLR_BG_TRAIL  = (0,   180,   0)   # green trail    – background flow
CLR_OBS_TRAIL = (0,     0, 255)   # red trail      – obstacle flow
CLR_OBS_RING  = (0,     0, 255)   # red ring       – repulsive warning
CLR_SNAP      = (100, 100, 100)   # grey cross     – snapped-back node
CLR_EM        = (0,   180, 255)   # orange         – emergency force
# ─────────────────────────────────────────────────────────────────────────────

# ── Road-centre restoring force (smooth tanh curve) ──────────────────────────
# Force is ALWAYS active, never snaps on/off.
# Shape:   F_road = -ROAD_GAIN × tanh(pos[1] / ROAD_SOFTNESS)
#
#   ROAD_SOFTNESS  controls the width of the "comfortable" centre zone.
#                  Small value → steep curve, strong correction even near centre.
#                  Large value → flat near centre, only kicks in far from y=0.
#
#   ROAD_GAIN      overall magnitude of the restoring force.
#
#   ROAD_HEADING_GAIN  blends in goal-heading alignment scaled by lateral error,
#                      so the car curves back toward y=0 already pointing at goal.
#
ROAD_SOFTNESS      = 0.8   # metres – half-width of the gentle centre zone
ROAD_GAIN          = 1.4   # lateral restoring strength
ROAD_HEADING_GAIN  = 0.6   # heading blend (fraction of heading_err added)
# ─────────────────────────────────────────────────────────────────────────────

# ── Grid layout ───────────────────────────────────────────────────────────────
GRID_STEP   = 16   # pixels between adjacent nodes
GRID_MARGIN = 8    # pixels from image border
# ─────────────────────────────────────────────────────────────────────────────

# ── Obstacle detection ────────────────────────────────────────────────────────
FLOW_ANOMALY_K = 2.5   # obstacle if speed > K × median scene speed
MIN_OBS_SPEED  = 0.5   # minimum absolute speed (px/frame) to flag
# ─────────────────────────────────────────────────────────────────────────────


# ── HUD helpers ───────────────────────────────────────────────────────────────

def draw_force_arrow(img, origin, force_x, scale=60.0, color=(255, 255, 255),
                     label="", thickness=2, tip_length=0.35):
    ox, oy = int(origin[0]), int(origin[1])
    ex = int(ox + force_x * scale)
    cv2.arrowedLine(img, (ox, oy), (ex, oy), color, thickness,
                    tipLength=tip_length)
    if label:
        cv2.putText(img, f"{label}: {force_x:+.2f}",
                    (ox + 4, oy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def draw_force_panel(rgb, f_att, f_rep, f_em, f_total, H, W):
    panel_h = 94        # taller to fit 4 rows
    y0      = H - panel_h
    overlay = rgb.copy()
    cv2.rectangle(overlay, (0, y0), (W, H), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, rgb, 0.45, 0, rgb)

    cx      = W // 2
    row_gap = 21
    cv2.line(rgb, (cx, y0 + 2), (cx, H - 2), (80, 80, 80), 1)

    for i, (fx, lbl, clr) in enumerate(zip(
            [f_att,   f_rep,   f_em,   f_total],
            ["F_att", "F_rep", "F_em", "F_tot"],
            [CLR_ATT, CLR_REP, CLR_EM, CLR_TOT])):
        ry = y0 + 10 + row_gap * i
        draw_force_arrow(rgb, (cx, ry), fx, scale=60.0, color=clr, label=lbl)
        cv2.circle(rgb, (W - 72, ry), 4, clr, -1)


# ─────────────────────────────────────────────────────────────────────────────

class FlowControllerGrid:
    """
    Tracks a permanent fixed-coordinate grid.

    grid_home  (N, 2) float32 – the immutable pixel address of every node.

    Each frame:
      1. Feed p0  (current live positions, always N points) into LK.
      2. Get p1 + status back from LK.
      3. For every node where status == 0 (lost):
             new_positions[i] = grid_home[i]   ← exact pixel snap
      4. Compute flow vectors only on successfully tracked nodes.
      5. Obstacle classification purely from flow-speed anomaly.
    """

    def __init__(self, car_id, steer_joints, motor_joints):
        self.car_id       = car_id
        self.steer_joints = steer_joints
        self.motor_joints = motor_joints

        self.tracker = LucasKanadeTracker_1(win=31, levels=3,
                                            max_iter=20, eps=0.03)
        self.W, self.H = 320, 240

        # ── Build permanent grid ──────────────────────────────────────────
        xs = np.arange(GRID_MARGIN, self.W - GRID_MARGIN, GRID_STEP,
                       dtype=np.float32)
        ys = np.arange(GRID_MARGIN, self.H - GRID_MARGIN, GRID_STEP,
                       dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)

        # grid_home : (N, 2)  – canonical addresses, never mutated
        self.grid_home = np.column_stack(
            (xv.ravel(), yv.ravel())).astype(np.float32)
        self.N = len(self.grid_home)

        # p0 : (N, 2)  – live tracked positions, starts at home
        self.p0        = self.grid_home.copy()
        self.prev_gray = None

        print(f"[Grid] {self.N} permanent nodes  "
              f"({xs.size} cols × {ys.size} rows,  step={GRID_STEP} px)")

    # ── Camera ────────────────────────────────────────────────────────────────

    def get_camera_frame(self):
        pos, orn   = p.getBasePositionAndOrientation(self.car_id)
        rot_mat    = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        camera_pos = pos + rot_mat.dot([0.4, 0.0, 0.35])
        target_pos = pos + rot_mat.dot([2.0, 0.0, 0.35])

        view_mat = p.computeViewMatrix(camera_pos, target_pos, [0, 0, 1])
        proj_mat = p.computeProjectionMatrixFOV(
            fov=60, aspect=self.W / self.H, nearVal=0.1, farVal=100)

        img_arr = p.getCameraImage(self.W, self.H, view_mat, proj_mat,
                                   renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb  = img_arr[2][:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return gray, rgb.copy()

    # ── Obstacle classification ───────────────────────────────────────────────

    def classify_obstacles(self, speeds):
        """
        Returns bool array of length M (tracked nodes only).
        A node is an obstacle if its flow speed is anomalously large
        compared to the median of all currently tracked nodes.

        Using median:
          - Background nodes (road, sky, far walls) dominate numerically
            and keep the median low.
          - A looming obstacle cluster gets flagged without needing any
            depth or segmentation information.
        """
        median_spd = np.median(speeds)
        threshold  = max(FLOW_ANOMALY_K * median_spd, MIN_OBS_SPEED)
        return speeds > threshold

    # ── FOE ───────────────────────────────────────────────────────────────────

    def compute_foe(self, p0, p1):
        v  = p1 - p0
        x,  y  = p0[:, 0], p0[:, 1]
        vx, vy = v[:, 0],  v[:, 1]
        A = np.column_stack((vy, -vx))
        b = x * vy - y * vx
        try:
            foe, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return foe
        except Exception:
            return np.array([self.W / 2, self.H / 2])

    # ── Main loop ─────────────────────────────────────────────────────────────

    def navigate(self, target_speed=10.0):
        print("[Grid] Running – press Q to quit")
        self.prev_gray, _ = self.get_camera_frame()

        while True:
            curr_gray, rgb = self.get_camera_frame()

            # Constant forward drive
            p.setJointMotorControlArray(
                self.car_id, self.motor_joints, p.VELOCITY_CONTROL,
                targetVelocities=[target_speed] * len(self.motor_joints),
                forces=[150] * len(self.motor_joints))

            F_att_x = F_rep_x = F_em_x = F_total_x = 0.0

            # ── LK pass on ALL N grid nodes ───────────────────────────────
            # p0 is always shape (N, 1, 2) – the full grid, every frame
            p1_lk, status = self.tracker.track(
                self.prev_gray, curr_gray,
                self.p0.reshape(-1, 1, 2).astype(np.float32))

            p1_flat  = p1_lk.reshape(-1, 2)           # (N, 2)
            status_f = status.flatten().astype(bool)   # (N,)  True = tracked OK

            # ── Snap lost nodes exactly back to their home pixel ──────────
            new_positions             = p1_flat.copy()
            new_positions[~status_f]  = self.grid_home[~status_f]
            #   ↑  This is the key line: lost node i → grid_home[i] exactly.
            #   No search, no nearest-neighbour – straight assignment.

            # ── Flow on successfully tracked nodes only ───────────────────
            tracked_idx = np.where(status_f)[0]       # indices into (N,)

            if len(tracked_idx) > 10:
                old_pts = self.p0[tracked_idx]         # (M, 2)
                new_pts = new_positions[tracked_idx]   # (M, 2)

                foe  = self.compute_foe(old_pts, new_pts)

                v      = new_pts - old_pts
                vx, vy = v[:, 0], v[:, 1]
                x,  y  = old_pts[:, 0], old_pts[:, 1]
                speeds = np.sqrt(vx**2 + vy**2) + 1e-5

                dist_foe = np.sqrt((x - foe[0])**2 + (y - foe[1])**2)
                ttc      = dist_foe / speeds
                looming  = ((x - foe[0]) * vx + (y - foe[1]) * vy) > 0

                is_obstacle = self.classify_obstacles(speeds)  # (M,) bool

                # ── Visualise every grid node ─────────────────────────────
                # Snapped (lost) nodes → grey cross at their home pixel
                for gi in np.where(~status_f)[0]:
                    gx = int(self.grid_home[gi, 0])
                    gy = int(self.grid_home[gi, 1])
                    cv2.drawMarker(rgb, (gx, gy), CLR_SNAP,
                                   cv2.MARKER_CROSS, 5, 1)

                # Tracked nodes → coloured dot + flow trail
                for ii, gi in enumerate(tracked_idx):
                    nx, ny = int(new_pts[ii, 0]), int(new_pts[ii, 1])
                    ox, oy = int(old_pts[ii, 0]), int(old_pts[ii, 1])
                    obs    = is_obstacle[ii]

                    cv2.circle(rgb, (nx, ny), 2,
                               CLR_OBS_DOT if obs else CLR_BG_DOT, -1)
                    cv2.line(rgb, (nx, ny), (ox, oy),
                             CLR_OBS_TRAIL if obs else CLR_BG_TRAIL, 1)

                    # Repulsive force: obstacle + looming + close TTC
                    if (obs
                            and looming[ii]
                            and ttc[ii] < 20.0
                            and y[ii] > self.H * 0.3):
                        direction = -1 if x[ii] > self.W / 2 else 1
                        F_rep_x  += direction * (50.0 / (ttc[ii] + 1e-2))
                        cv2.circle(rgb, (int(x[ii]), int(y[ii])),
                                   7, CLR_OBS_RING, 2)

                # ── Attractive force ──────────────────────────────────────
                pos, orn    = p.getBasePositionAndOrientation(self.car_id)
                yaw         = p.getEulerFromQuaternion(orn)[2]
                theta_goal  = np.arctan2(0.0 - pos[1], 31.66 - pos[0])
                heading_err = (theta_goal - yaw + np.pi) % (2 * np.pi) - np.pi
                F_att_x     = 0.8 * heading_err

                # ── Road-centre restoring force (smooth curve) ────────────
                #
                # F_road is a tanh potential well centred at y = 0.
                # It is ALWAYS active – no threshold, no snap.
                #
                #   tanh(pos[1] / ROAD_SOFTNESS)
                #     → ≈ pos[1]/ROAD_SOFTNESS  near the centre  (linear, gentle)
                #     → ≈ ±1                    far from centre  (saturates, firm)
                #
                # Multiplied by -ROAD_GAIN so the force always points inward.
                #
                # A heading component scaled by |pos[1]| / ROAD_SOFTNESS is
                # blended in: near the centre it is negligible; far out it
                # amplifies goal-alignment so the car re-enters pointing forward.
                #
                lateral_err   = pos[1]                          # signed offset from y=0
                road_curve    = np.tanh(lateral_err / ROAD_SOFTNESS)
                F_lateral     = -ROAD_GAIN * road_curve
                heading_blend = ROAD_HEADING_GAIN * abs(road_curve) * heading_err
                F_em_x        = F_lateral + heading_blend

                # HUD tint: intensity proportional to |road_curve| so it fades
                # gracefully as the car approaches centre
                tint_alpha = float(np.clip(abs(road_curve) - 0.15, 0, 1))
                if tint_alpha > 0:
                    tint = rgb.copy()
                    tint[:, :, 0] = np.clip(tint[:, :, 0].astype(int)
                                            + int(60 * tint_alpha), 0, 255)
                    cv2.addWeighted(tint, tint_alpha * 0.35, rgb,
                                    1 - tint_alpha * 0.35, 0, rgb)
                    cv2.putText(rgb, f"lane y={lateral_err:+.2f}",
                                (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (0, 180, 255), 1, cv2.LINE_AA)

                # ── Steer ─────────────────────────────────────────────────
                F_total_x   = F_att_x + 0.1 * F_rep_x + F_em_x
                steer_angle = np.clip(F_total_x, -0.6, 0.6)
                p.setJointMotorControlArray(
                    self.car_id, self.steer_joints, p.POSITION_CONTROL,
                    targetPositions=[steer_angle] * len(self.steer_joints))

                # FOE marker
                cv2.circle(rgb, (int(foe[0]), int(foe[1])), 8, CLR_FOE, -1)
                cv2.putText(rgb, "FOE",
                            (int(foe[0]) + 10, int(foe[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_FOE, 2)

            # ── Commit positions (lost nodes already snapped) ─────────────
            self.p0 = new_positions

            # ── HUD ───────────────────────────────────────────────────────
            draw_force_panel(rgb, F_att_x, F_rep_x, F_em_x, F_total_x, self.H, self.W)

            cv2.imshow("Flow Controller [Fixed Grid]",
                       cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.prev_gray = curr_gray.copy()
            p.stepSimulation()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    result = setup_simulation(gui=True)
    car_id, steer_j, motor_j = result[:3]
    controller = FlowControllerGrid(car_id, steer_j, motor_j)
    controller.navigate()