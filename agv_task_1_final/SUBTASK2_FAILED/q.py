import cv2
import numpy as np
import pybullet as p
import time

from simulation_setup import setup_simulation

try:
    from lucas_kanade import LucasKanadeTracker_1 as LKTracker
except ImportError:
    from lucas_kanade import LucasKanadeTracker as LKTracker


class VisualPotentialFieldController:
    def __init__(self, width=640, height=480):
        self.W = width
        self.H = height
        self.goal = np.array([self.W / 2, self.H / 2])
        
        # Controller tuning parameters
        self.alpha = 0.2       # Attractive force gain
        self.gamma = 50.0      # Repulsive force gain
        
        # Road Boundaries
        self.A_rd = 0.5        
        self.b_rd = 1.0        
        self.y_r = 1.16        # Right road edge 
        self.y_l = -1.16       # Left road edge 
        
        self.K_steer_img = 0.8 # Image forces to steering mapping
        self.K_steer_rd = 1.2  # Road forces to steering mapping
        
        # State tracking
        self.prev_gray = None
        self.prev_pts = None
        self.current_steer = 0.0 
        
        self.tracker = LKTracker(win=51, levels=3, max_iter=20, eps=0.03)

    def get_camera_image(self, car_id):
        """Extracts the ego-perspective camera frame from PyBullet."""
        car_pos, car_orn = p.getBasePositionAndOrientation(car_id)
        rot_matrix = np.array(p.getMatrixFromQuaternion(car_orn)).reshape(3, 3)
        
        cam_pos = car_pos + rot_matrix @ np.array([1.2, 0.0, 0.4])
        target_pos = cam_pos + rot_matrix @ np.array([1.0, 0.0, 0.0])
        up_vec = rot_matrix @ np.array([0.0, 0.0, 1.0])
        
        view_matrix = p.computeViewMatrix(cam_pos, target_pos, up_vec)
        proj_matrix = p.computeProjectionMatrixFOV(60, self.W / self.H, 0.1, 100)
        
        _, _, img, _, _ = p.getCameraImage(
            self.W, self.H, view_matrix, proj_matrix, 
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        frame = np.reshape(img, (self.H, self.W, 4))[:, :, :3].astype(np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame, gray

    def compute_foe_and_ttc(self, pts1, pts2, status):
        """Calculates Focus of Expansion and TTC per tracked feature point."""
        good_new = pts2[status.flatten() == 1].reshape(-1, 2)
        good_old = pts1[status.flatten() == 1].reshape(-1, 2)
        
        if len(good_new) < 5:
            return self.goal, [], [], []

        vx = good_new[:, 0] - good_old[:, 0]
        vy = good_new[:, 1] - good_old[:, 1]
        x = good_old[:, 0]
        y = good_old[:, 1]
        
        A = np.column_stack((vy, -vx))
        b = x * vy - y * vx
        
        try:
            FOE = np.linalg.pinv(A.T @ A + np.eye(2)*1e-2) @ A.T @ b
            FOE[0] = np.clip(FOE[0], -self.W, 2*self.W)
            FOE[1] = np.clip(FOE[1], -self.H, 2*self.H)
        except np.linalg.LinAlgError:
            FOE = self.goal
            
        dist_to_foe = np.sqrt((x - FOE[0])**2 + (y - FOE[1])**2)
        magnitude = np.sqrt(vx**2 + vy**2)
        ttc = dist_to_foe / (magnitude + 1e-5)
        
        return FOE, good_old, ttc, magnitude

    def calculate_forces(self, FOE, points, ttcs, magnitudes, car_y):
        """Computes Attractive, Repulsive, and Road Potential Forces."""
        F_att_x = self.alpha * (self.goal[0] - FOE[0]) / self.W
        F_att_y = self.alpha * (self.goal[1] - FOE[1]) / self.H
        
        F_rep_x, F_rep_y = 0.0, 0.0
        obs_pts = 0
        min_ttc = 999.0
        
        for i in range(len(points)):
            x_i, y_i = points[i]
            
            # CORE FIX: Narrowed the horizontal vision to ignore passing lane lines, 
            # and dropped magnitude to 0.05 to catch dead-center obstacles[cite: 955].
            if ttcs[i] < 12.0 and magnitudes[i] > 0.05 and y_i < self.H * 0.85 and (self.W * 0.15 < x_i < self.W * 0.85):
                min_ttc = min(min_ttc, ttcs[i])
                
                dx = FOE[0] - x_i
                dy = FOE[1] - y_i
                
                # BREAK SYMMETRY: Force a hard swerve if an obstacle is directly in front
                if abs(dx) < 80.0 and ttcs[i] < 6.0:
                    dx = -150.0 if x_i > self.W/2 else 150.0
                    
                norm = np.sqrt(dx**2 + dy**2) + 1e-5
                weight = 1.0 / (ttcs[i] + 0.1)
                
                F_rep_x += (dx / norm) * weight
                F_rep_y += (dy / norm) * weight
                obs_pts += 1
                
        if obs_pts > 0:
            F_rep_x = self.gamma * (F_rep_x / obs_pts)
            F_rep_y = self.gamma * (F_rep_y / obs_pts)
            
        # Road Potential Field -> Proportional Lane Keeping Assist
        F_road_x = car_y * 1.5  
        
        if car_y > 0.7: 
            F_road_x += 2.5 / max(self.y_r - car_y, 0.01) 
        elif car_y < -0.7: 
            F_road_x -= 2.5 / max(car_y - self.y_l, 0.01) 

        F_tot_x = F_att_x + F_rep_x
        F_tot_y = F_att_y + F_rep_y
        
        return F_tot_x, F_tot_y, F_rep_x, F_rep_y, F_att_x, F_att_y, F_road_x, obs_pts, min_ttc

    def render_hud(self, frame, FOE, forces, stats):
        """Draws the diagnostic overlay."""
        F_tot_x, F_tot_y, F_rep_x, F_rep_y, F_att_x, F_att_y, F_road_x = forces
        obs_pts, total_pts, min_ttc, speed, steer = stats
        
        CG = (int(self.W/2), int(self.H * 0.8))
        
        foe_pt = (int(FOE[0]), int(FOE[1]))
        cv2.drawMarker(frame, foe_pt, (0, 215, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.circle(frame, foe_pt, 10, (0, 215, 255), 2)
        cv2.putText(frame, "FOE", (foe_pt[0]-15, foe_pt[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 2)
        
        scale = 50
        cv2.arrowedLine(frame, CG, (CG[0] + int(F_att_x * scale * 5), CG[1] + int(F_att_y * scale * 5)), (0, 255, 0), 2)
        cv2.arrowedLine(frame, CG, (CG[0] + int(F_rep_x * scale), CG[1] + int(F_rep_y * scale)), (0, 0, 255), 2)
        cv2.arrowedLine(frame, CG, (CG[0] + int((F_tot_x + F_road_x) * scale), CG[1] + int(F_tot_y * scale)), (255, 255, 0), 2)
        
        lines = [
            f"F_total : ({F_tot_x + F_road_x:+.2f}, {F_tot_y:+.2f})",
            f"F_rep   : ({F_rep_x:+.2f}, {F_rep_y:+.2f})",
            f"F_att   : ({F_att_x:+.2f}, {F_att_y:+.2f})",
            f"F_road  : ({F_road_x:+.2f},  N/A)",
            f"Obstacles: {obs_pts} / {total_pts} pts",
            f"Min TTC  : {min_ttc:.2f} s",
            f"Speed    : {speed:.3f}",
            f"Steering : {steer:+.3f}"
        ]
        
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        return frame


def run():
    car_id, steer_j, motor_j = setup_simulation()
    controller = VisualPotentialFieldController()
    
    # Requesting 400 features so we have plenty to track on both the road and the obstacles
    feature_params = dict(maxCorners=400, qualityLevel=0.01, minDistance=10, blockSize=7)

    print("\n--- Optical Flow Controller Started ---")
    while True:
        car_pos, car_orn = p.getBasePositionAndOrientation(car_id)
        car_vel, _ = p.getBaseVelocity(car_id)
        speed = np.linalg.norm(car_vel)
        car_local_y = car_pos[1]
        
        frame, gray = controller.get_camera_image(car_id)
        
        # CORE FIX: Increased redetection threshold from 50 to 250. 
        # This guarantees that as lane lines pass out of view, we aggressively scan for new obstacles!
        if controller.prev_pts is None or len(controller.prev_pts) < 250:
            controller.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
            if controller.prev_pts is not None:
                controller.prev_pts = controller.prev_pts.reshape(-1, 1, 2)
            controller.prev_gray = gray.copy()
            continue
            
        next_pts, status = controller.tracker.track(controller.prev_gray, gray, controller.prev_pts)
        
        FOE, points, ttcs, magnitudes = controller.compute_foe_and_ttc(controller.prev_pts, next_pts, status)
        
        F_tot_x, F_tot_y, F_rep_x, F_rep_y, F_att_x, F_att_y, F_road_x, obs_pts, min_ttc = controller.calculate_forces(
            FOE, points, ttcs, magnitudes, car_local_y
        )
        
        target_steering_cmd = -(controller.K_steer_img * F_tot_x + controller.K_steer_rd * F_road_x)
        target_steering_cmd = np.clip(target_steering_cmd, -0.5, 0.5) 
        
        controller.current_steer = (0.7 * controller.current_steer) + (0.3 * target_steering_cmd)
        
        target_speed = 5.55
        if min_ttc < 2.5 and obs_pts > 3:
            target_speed = 1.5
        elif min_ttc < 1.0 and obs_pts > 3:
            target_speed = -2.0 # Brake!
            
        for j in steer_j:
            p.setJointMotorControl2(car_id, j, p.POSITION_CONTROL, targetPosition=controller.current_steer)
        for j in motor_j:
            p.setJointMotorControl2(car_id, j, p.VELOCITY_CONTROL, targetVelocity=target_speed, force=1000)
            
        forces = (F_tot_x, F_tot_y, F_rep_x, F_rep_y, F_att_x, F_att_y, F_road_x)
        stats = (obs_pts, len(points), min_ttc, speed, controller.current_steer)
        hud_frame = controller.render_hud(frame.copy(), FOE, forces, stats)
        
        for i, pt in enumerate(points):
            # Visually verify that the points on the box are triggering correctly
            color = (0, 0, 255) if (ttcs[i] < 12.0 and magnitudes[i] > 0.05 and points[i][1] < controller.H * 0.85 and (controller.W * 0.15 < points[i][0] < controller.W * 0.85)) else (0, 255, 0)
            cv2.circle(hud_frame, (int(pt[0]), int(pt[1])), 3, color, -1)
            
        cv2.imshow("Flow Controller", hud_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        controller.prev_gray = gray.copy()
        controller.prev_pts = next_pts[status.flatten() == 1].reshape(-1, 1, 2)
        p.stepSimulation()
        time.sleep(1./60.)

    p.disconnect()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()