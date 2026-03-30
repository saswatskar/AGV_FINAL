"""
flow_controller_grid.py
────────────────────────────────────────────────────────────────────────────
Feature strategy : Uniform pixel grid  (fixed spatial sampling)
Obstacle detection: Flow-speed anomaly – points whose optical-flow magnitude
                    significantly exceeds the scene median are treated as
                    lying on near/looming obstacles.

Grid approach explained
───────────────────────
Instead of letting Shi-Tomasi pick high-contrast corners, we seed a fixed
lattice of (row, col) sample points that tiles the whole image uniformly.
Benefits
  • Homogeneous coverage – wide, low-texture obstacles (e.g. a white wall)
    are sampled even when they contain no strong corners.
  • Predictable density – the number of active points stays nearly constant;
    no dependency on scene texture richness.
Trade-off
  • LK can drift more on flat regions between refreshes; we counter this by
    resetting dead/drifted points back to their grid home every N frames.
"""

import cv2
import numpy as np
import pybullet as p
import time

from lucas_kanade import LucasKanadeTracker_1
from simulation_setup import setup_simulation


# ── Colour palette ────────────────────────────────────────────────────────────
CLR_ATT      = (0,   220,   0)
CLR_REP      = (0,    60, 255)
CLR_TOT      = (0,   220, 255)
CLR_FOE      = (255,   0,   0)
CLR_FLOW_BG  = (0,   180,   0)
CLR_FLOW_OBS = (0,     0, 255)
CLR_OBS_RING = (0,     0, 255)
# ─────────────────────────────────────────────────────────────────────────────
 
# ── Road boundary ─────────────────────────────────────────────────────────────
ROAD_HALF_WIDTH = 0.5   # |pos[1]| threshold beyond which emergency force fires
EM_LATERAL_GAIN = 2   # how hard to push laterally back to centre
EM_HEADING_GAIN = 1.5   # extra heading-alignment gain applied during emergency
# ─────────────────────────────────────────────────────────────────────────────
 

# ── Grid hyper-parameters ─────────────────────────────────────────────────────
GRID_STEP       = 15    # pixels between adjacent grid nodes
GRID_MARGIN     = 10    # pixel margin from image border
GRID_REFRESH_N  = 20    # re-seed dead grid points every N frames
MAX_DRIFT_PX    = 20.0  # if a point drifts > this far from its home → reset
# ─────────────────────────────────────────────────────────────────────────────

# ── Obstacle-detection hyper-parameters ──────────────────────────────────────
FLOW_ANOMALY_K = 1.6
MIN_OBS_SPEED  = 0.3
# ─────────────────────────────────────────────────────────────────────────────
RADIAL_DOT_THRESH = 0.3   # cos angle; below this = not background-like


# ── HUD helpers ──────────────────────────────────────────────────────────────

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


def draw_force_panel(rgb, f_att, f_rep, f_total, H, W):
    panel_h = 72
    y0      = H - panel_h

    overlay = rgb.copy()
    cv2.rectangle(overlay, (0, y0), (W, H), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, rgb, 0.45, 0, rgb)

    cx      = W // 2
    row_gap = 22
    cv2.line(rgb, (cx, y0 + 2), (cx, H - 2), (80, 80, 80), 1)

    rows   = [y0 + 10, y0 + 10 + row_gap, y0 + 10 + row_gap * 2]
    forces = [f_att,   f_rep,             f_total]
    labels = ["F_att", "F_rep",           "F_tot"]
    colors = [CLR_ATT, CLR_REP,           CLR_TOT]

    for ry, fx, lbl, clr in zip(rows, forces, labels, colors):
        draw_force_arrow(rgb, (cx, ry), fx, scale=60.0, color=clr, label=lbl)
        cv2.circle(rgb, (W - 72, ry), 4, clr, -1)


# ─────────────────────────────────────────────────────────────────────────────

class FlowControllerGrid:
    """
    Tracks a uniform grid of points.  Obstacle classification is purely
    flow-speed based – no segmentation required.
    """

    def __init__(self, car_id, steer_joints, motor_joints):
        self.car_id       = car_id
        self.steer_joints = steer_joints
        self.motor_joints = motor_joints

        self.tracker   = LucasKanadeTracker_1(win=31, levels=3,
                                              max_iter=20, eps=0.03)
        self.prev_gray = None

        self.W, self.H  = 320, 240
        self.frame_idx  = 0

        # Build the static grid of home positions and seed the live points
        self.grid_home  = self._build_grid()   # shape (N, 2)  [x, y]
        self.p0         = self.grid_home.reshape(-1, 1, 2).astype(np.float32)

    # ── Grid construction ─────────────────────────────────────────────────────

    def _build_grid(self):
        """
        Returns an (N, 2) array of (x, y) coordinates spaced GRID_STEP apart,
        leaving GRID_MARGIN pixels of border on each side.
        """
        xs = np.arange(GRID_MARGIN, self.W - GRID_MARGIN, GRID_STEP)
        ys = np.arange(GRID_MARGIN, self.H - GRID_MARGIN, GRID_STEP)
        xv, yv = np.meshgrid(xs, ys)
        grid = np.column_stack((xv.ravel(), yv.ravel())).astype(np.float32)
        return grid

    # ── Grid maintenance ──────────────────────────────────────────────────────

    def _refresh_grid_points(self, active_pts):
        """
        For every home node whose nearest tracked point has drifted more than
        MAX_DRIFT_PX, re-insert the home node into the active set.

        active_pts : (N, 2) current tracked positions
        returns    : (M, 1, 2) merged float32 array
        """
        if active_pts is None or len(active_pts) == 0:
            return self.grid_home.reshape(-1, 1, 2).astype(np.float32)

        # For each home node, find distance to closest active point
        diffs  = active_pts[:, None, :] - self.grid_home[None, :, :]  # (N,G,2)
        dists  = np.sqrt((diffs ** 2).sum(axis=2))                     # (N,G)
        min_d  = dists.min(axis=0)                                     # (G,)

        # Homes that are uncovered
        orphans = self.grid_home[min_d > MAX_DRIFT_PX]                 # (k, 2)

        if len(orphans) == 0:
            return active_pts.reshape(-1, 1, 2).astype(np.float32)

        merged = np.vstack([active_pts, orphans])
        return merged.reshape(-1, 1, 2).astype(np.float32)

    # ── Camera ────────────────────────────────────────────────────────────────

    def get_camera_frame(self):
        pos, orn   = p.getBasePositionAndOrientation(self.car_id)
        rot_mat    = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        camera_pos = pos + rot_mat.dot([0.4, 0.0, 0.35])
        target_pos = pos + rot_mat.dot([2.0, 0.0, 0.35])

        view_mat = p.computeViewMatrix(camera_pos, target_pos, [0, 0, 1])
        proj_mat = p.computeProjectionMatrixFOV(
            fov=90, aspect=self.W / self.H, nearVal=0.1, farVal=100)

        img_arr = p.getCameraImage(self.W, self.H, view_mat, proj_mat,
                                   renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb  = img_arr[2][:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return gray, rgb.copy()

    # ── Obstacle detection ────────────────────────────────────────────────────

    def classify_obstacle_points(self, speeds):
        """
        True where flow speed is anomalously high compared to scene median.

        Why median?
        ───────────
        With a grid, many background points sit on road/sky and have low,
        consistent speeds. A looming obstacle produces a cluster of fast
        points. The median stays anchored in the background majority no
        matter how large the obstacle cluster is (up to ~50 % of points).
        """
        median_speed = np.median(speeds)
        threshold    = max(FLOW_ANOMALY_K * median_speed, MIN_OBS_SPEED)
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
        print("[Grid] Starting – press Q to quit")
        self.prev_gray, _ = self.get_camera_frame()

        while True:
            self.frame_idx += 1
            curr_gray, rgb  = self.get_camera_frame()

            p.setJointMotorControlArray(
                self.car_id, self.motor_joints, p.VELOCITY_CONTROL,
                targetVelocities=[target_speed] * len(self.motor_joints),
                forces=[150] * len(self.motor_joints))

            F_att_x = F_rep_x = F_total_x = 0.0

            if self.p0 is not None and len(self.p0) > 15:
                p1, status = self.tracker.track(self.prev_gray, curr_gray, self.p0)
                good_new   = p1[status.flatten() == 1].reshape(-1, 2)
                good_old   = self.p0[status.flatten() == 1].reshape(-1, 2)

                if len(good_new) > 10:
                    foe = self.compute_foe(good_old, good_new)

                    v          = good_new - good_old
                    vx, vy     = v[:, 0], v[:, 1]
                    x,  y      = good_old[:, 0], good_old[:, 1]
                    speeds     = np.sqrt(vx**2 + vy**2) + 1e-5

                    dist_foe   = np.sqrt((x - foe[0])**2 + (y - foe[1])**2)
                    ttc        = dist_foe / speeds

                    looming    = ((x - foe[0]) * vx + (y - foe[1]) * vy) > 0
# ── Flow-anomaly obstacle classification ──────────────
                    is_obstacle = self.classify_obstacle_points(speeds)

                    # Initialize Local Frame Forces (X = Longitudinal/Forward, Y = Lateral/Left)
                    F_obs_X = 0.0
                    F_obs_Y = 0.0
                    gamma_obs = 10000.0 # Gain for repulsive obstacle force (Section IV-B)

                    for i in range(len(good_new)):
                        nx, ny     = int(good_new[i, 0]), int(good_new[i, 1])
                        ox_pt, oy_pt = int(good_old[i, 0]), int(good_old[i, 1])

                        trail_clr = CLR_FLOW_OBS if is_obstacle[i] else CLR_FLOW_BG
                        cv2.line(rgb, (nx, ny), (ox_pt, oy_pt), trail_clr, 1)

                        cv2.circle(rgb, (nx, ny), 2,
                                   CLR_FLOW_OBS if is_obstacle[i] else (40, 120, 40), -1)

                        radial_vec  = np.array([x[i] - foe[0], y[i] - foe[1]])
                        flow_vec    = np.array([vx[i], vy[i]])
                        norm_r      = np.linalg.norm(radial_vec) + 1e-5
                        norm_f      = np.linalg.norm(flow_vec)   + 1e-5
                        dot_radial  = np.dot(radial_vec, flow_vec) / (norm_r * norm_f)

                        not_background = dot_radial < RADIAL_DOT_THRESH

                        # ── 1. Obstacle Potential Field (Section IV-B, Eq 9) ──
                        if (is_obstacle[i]
                                and looming[i]
                                and not_background
                                and ttc[i] < 20.0
                                and y[i] > self.H * 0.3):
                            
                            # Approximate 3D projection gradient direction mapping to lateral (Y)
                            direction_y = 1 if x[i] < foe[0] else -1 
                            
                            # Repulsive force components based on TTC mapping
                            F_obs_Y += direction_y * gamma_obs / (ttc[i] + 1e-2)
                            F_obs_X -= gamma_obs / (ttc[i] + 1e-2) # Slow down effect

                            cv2.circle(rgb, (int(x[i]), int(y[i])), 6, CLR_OBS_RING, 2)

                    # ── 2. Target Potential Field (Section IV-A, Eq 7 & 8) ──
                    pos, orn    = p.getBasePositionAndOrientation(self.car_id)
                    yaw         = p.getEulerFromQuaternion(orn)[2]
                    goal_x, goal_y = 31.66, 0.0
                    
                    dist_goal   = np.sqrt((goal_x - pos[0])**2 + (goal_y - pos[1])**2)
                    theta_goal  = np.arctan2(goal_y - pos[1], goal_x - pos[0])
                    
                    alpha_att   = 1.0 # Attraction scaling constant
                    F_att       = alpha_att * dist_goal
                    
                    # Transform goal force gradient into local vehicle frame
                    F_att_X     = F_att * np.cos(theta_goal - yaw)
                    F_att_Y     = F_att * np.sin(theta_goal - yaw)

                    # ── 3. Road Potential Field (Section IV-C, Eq 10, 11 & 18) ──
                    y_current = pos[1] 
                    y_l = ROAD_HALF_WIDTH   # Left lane boundary (+Y in PyBullet)
                    y_r = -ROAD_HALF_WIDTH  # Right lane boundary (-Y in PyBullet)
                    
                    A_rd = 0.5
                    # Parameter 'b' tuned up from Table I to fit the narrower simulation road scale
                    b_rd = 8.0 
                    
                    # Partial derivatives of Modified Morse Potential U_s w.r.t lateral position
                    dU_sr_dy = 2 * A_rd * b_rd * np.exp(-b_rd * (y_current - y_r)) * (1 - np.exp(-b_rd * (y_current - y_r)))
                    dU_sl_dy = -2 * A_rd * b_rd * np.exp(b_rd * (y_current - y_l)) * (1 - np.exp(b_rd * (y_current - y_l)))
                    
                    # Total Repulsive road force is the negative gradient
                    F_road_Y = -(dU_sr_dy + dU_sl_dy)

                    # ── 4. Total Potential Field & Controller (Sections V & VI) ──
                    # Total forces mapped to motion plane (Eq 19 & 20)
                    lambda_X, lambda_Y = 1.0, 1.0 
                    F_total_X = F_att_X + lambda_X * F_obs_X
                    F_total_Y = F_att_Y + lambda_Y * F_obs_Y + F_road_Y

                    # Desired orientation from artificial gradient (Eq 27)
                    psi_d = np.arctan2(F_total_Y, max(F_total_X, 0.1)) # Max prevents backward division
                    
                    # Gradient Tracking Sliding Mode mapping (Eq 28 - 30)
                    steer_angle = np.clip(psi_d, -0.6, 0.6)
                    
                    p.setJointMotorControlArray(
                        self.car_id, self.steer_joints, p.POSITION_CONTROL,
                        targetPositions=[steer_angle] * len(self.steer_joints))

                    # Map local Y forces back to HUD variables to keep the UI panel working
                    F_att_x = F_att_Y
                    F_rep_x = F_obs_Y + F_road_Y
                    F_total_x = F_total_Y

                    # FOE Visualizer
                    cv2.circle(rgb, (int(foe[0]), int(foe[1])), 8, CLR_FOE, -1)
                    cv2.putText(rgb, "FOE", (int(foe[0]) + 10, int(foe[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_FOE, 2)

                # ── Periodic grid re-seeding ──────────────────────────────
                if self.frame_idx % GRID_REFRESH_N == 0:
                    self.p0 = self._refresh_grid_points(good_new)
                else:
                    self.p0 = good_new.reshape(-1, 1, 2).astype(np.float32)

            else:
                # Full reset to grid home positions
                self.p0 = self.grid_home.reshape(-1, 1, 2).astype(np.float32)

            draw_force_panel(rgb, F_att_x, F_rep_x, F_total_x, self.H, self.W)

            cv2.imshow("Flow Controller [Grid]",
                       cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.prev_gray = curr_gray.copy()
            p.stepSimulation()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    result   = setup_simulation(gui=True)
    car_id, steer_j, motor_j = result[:3]
    controller = FlowControllerGrid(car_id, steer_j, motor_j)
    controller.navigate()