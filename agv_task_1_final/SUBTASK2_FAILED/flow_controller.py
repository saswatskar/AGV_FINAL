import cv2
import numpy as np
import pybullet as p
import time

# Import the provided modules
from lucas_kanade import LucasKanadeTracker
from simulation_setup import setup_simulation

class FlowController:
    def __init__(self, car_id, steer_joints, motor_joints):
        self.car_id = car_id
        self.steer_joints = steer_joints
        self.motor_joints = motor_joints
        
        # Initialize the custom LK tracker
        self.tracker = LucasKanadeTracker(win=31, levels=3, max_iter=20, eps=0.03)
        self.prev_gray = None
        self.p0 = None
        
        # Camera Resolution
        self.W, self.H = 320, 240

    def get_camera_frame(self):
        """Extracts RGB and Grayscale frames from the PyBullet agent camera."""
        pos, orn = p.getBasePositionAndOrientation(self.car_id)
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        
        # Mount the camera slightly high and forward on the car
        camera_pos = pos + rot_matrix.dot(np.array([0.1, 0.0, 0.35]))
        target_pos = pos + rot_matrix.dot(np.array([2.0, 0.0, 0.35]))
        
        view_matrix = p.computeViewMatrix(camera_pos, target_pos, [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=self.W/self.H, nearVal=0.1, farVal=100)
        
        img_arr = p.getCameraImage(self.W, self.H, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb = img_arr[2][:, :, :3] # Extract RGB channels
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        return gray, rgb.copy()

    def compute_foe(self, p0, p1):
        """Computes the Focus of Expansion (FOE) using Least Squares."""
        v = p1 - p0
        x, y = p0[:, 0], p0[:, 1]
        vx, vy = v[:, 0], v[:, 1]
        
        # Build A and b matrices for least squares
        A = np.column_stack((vy, -vx))
        b = x * vy - y * vx
        
        try:
            # Solve (A^T A) FOE = A^T b
            foe, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return foe
        except Exception:
            # Fallback to image center if matrix is singular (e.g. no movement)
            return np.array([self.W / 2, self.H / 2])

    def detect_features(self, gray_img):
        """Detects strong corners to track."""
        return cv2.goodFeaturesToTrack(gray_img, maxCorners=150, qualityLevel=0.05, minDistance=10)

    def navigate(self, target_speed=20.0):
        """Main pipeline: perceives environment, computes potential field, and steers."""
        print("Starting Flow Controller...")
        self.prev_gray, _ = self.get_camera_frame()
        self.p0 = self.detect_features(self.prev_gray)
        
        while True:
            curr_gray, rgb = self.get_camera_frame()
            
            # Maintain constant forward velocity
            p.setJointMotorControlArray(
                self.car_id, self.motor_joints,
                p.VELOCITY_CONTROL,
                targetVelocities=[target_speed] * len(self.motor_joints),
                forces=[150] * len(self.motor_joints)
            )

            if self.p0 is not None and len(self.p0) > 15:
                # Track points
                p1, status = self.tracker.track(self.prev_gray, curr_gray, self.p0)
                
                good_new = p1[status.flatten() == 1].reshape(-1, 2)
                good_old = self.p0[status.flatten() == 1].reshape(-1, 2)
                
                if len(good_new) > 10:
                    # 1. Compute Focus of Expansion
                    foe = self.compute_foe(good_old, good_new)
                    
                    # 2. Compute Flow Dynamics
                    v = good_new - good_old
                    vx, vy = v[:, 0], v[:, 1]
                    x, y = good_old[:, 0], good_old[:, 1]
                    
                    dist_foe = np.sqrt((x - foe[0])**2 + (y - foe[1])**2)
                    speed_flow = np.sqrt(vx**2 + vy**2) + 1e-5
                    
                    # 3. Time-to-Contact (TTC)
                    ttc = dist_foe / speed_flow
                    
                    # Looming Check: Are points moving outwards from the FOE?
                    looming = ((x - foe[0]) * vx + (y - foe[1]) * vy) > 0
                    
                    # 4. Generate Visual Potential Field
                    F_rep_x = 0.0
                    for i in range(len(ttc)):
                        # If an object is looming, getting close (TTC < threshold), and in the lower 2/3rds of view
                        if looming[i] and ttc[i] < 20.0 and y[i] > self.H * 0.3:
                            # Push in the opposite lateral direction of the obstacle
                            direction = -1 if x[i] > self.W / 2 else 1
                            F_rep_x += direction * (50.0 / (ttc[i] + 1e-2))
                            
                            # Draw warning circles on obstacles
                            cv2.circle(rgb, (int(x[i]), int(y[i])), 6, (0, 0, 255), 2)
                            
# --- REPLACED ATTRACTIVE FORCE LOGIC ---
                    # 1. Get current car position and yaw orientation
                    pos, orn = p.getBasePositionAndOrientation(self.car_id)
                    yaw = p.getEulerFromQuaternion(orn)[2]
                    
                    # 2. Define Goal Position (End wall from simulation_setup.py)
                    goal_x, goal_y = 31.66, 0.0
                    
                    # 3. Calculate angle to the goal
                    theta_goal = np.arctan2(goal_y - pos[1], goal_x - pos[0])
                    
                    # 4. Calculate heading error (normalize to [-pi, pi])
                    heading_error = (theta_goal - yaw + np.pi) % (2 * np.pi) - np.pi
                    
                    # 5. Attractive force is proportional to the heading error
                    alpha = 0.8  # Attraction gain (tune this to balance with F_rep_x)
                    F_att_x = alpha * heading_error
                    # ---------------------------------------
                    
                    # Gradient based control: Total lateral force dictates steering
                    F_total_x = F_att_x + 0.1 * F_rep_x
                    steer_angle = np.clip(F_total_x, -0.6, 0.6)
                    
                    # Apply steering command
                    p.setJointMotorControlArray(
                        self.car_id, self.steer_joints,
                        p.POSITION_CONTROL,
                        targetPositions=[steer_angle] * len(self.steer_joints)
                    )
                    
                    # 5. Visualization
                    for new, old in zip(good_new, good_old):
                        cv2.line(rgb, (int(new[0]), int(new[1])), (int(old[0]), int(old[1])), (0, 255, 0), 1)
                    
                    # Draw FOE
                    cv2.circle(rgb, (int(foe[0]), int(foe[1])), 8, (255, 0, 0), -1)
                    cv2.putText(rgb, "FOE", (int(foe[0])+10, int(foe[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                
                self.p0 = good_new.reshape(-1, 1, 2)
            else:
                # Re-detect if we lose too many points
                self.p0 = self.detect_features(curr_gray)
                
            # Periodically add new features to maintain a rich flow field
            if self.p0 is None or len(self.p0) < 60:
                new_feats = self.detect_features(curr_gray)
                if new_feats is not None:
                    self.p0 = new_feats
                    
            cv2.imshow("Optical Flow Controller", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.prev_gray = curr_gray.copy()
            p.stepSimulation()
            time.sleep(1.0 / 60.0)

if __name__ == "__main__":
    # Standard PyBullet setup from provided file
    car_id, steer_j, motor_j = setup_simulation(gui=True)
    
    # Initialize and run
    controller = FlowController(car_id, steer_j, motor_j)
    controller.navigate()