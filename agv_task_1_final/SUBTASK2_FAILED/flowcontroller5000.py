import numpy as np
import cv2
import pybullet as p
import time
import math

# Import your simulation setup
from simulation_setup import setup_simulation

# Import your custom tracker
try:
    from lucas_kanade import LucasKanadeTracker
    HAS_CUSTOM_TRACKER = True
except ImportError:
    HAS_CUSTOM_TRACKER = False
    print("WARNING: Could not import LucasKanadeTracker. Make sure the class name matches!")

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
K_ATTRACTIVE = 1.0     # Pull towards the goal (31.66, 0)
K_REPULSIVE  = 0.015   # Push away from obstacles
MAX_SPEED    = 10.0    # Base driving speed
GOAL_POS     = np.array([31.66, 0.0])
STUCK_VEL    = 0.1     # Velocity threshold to consider car "stuck"
BACKTRACK_DUR = 60     # Frames to reverse when stuck
# =============================================================================

class FlowController:
    def __init__(self, car_id, steering_joints, motor_joints):
        self.car_id = car_id
        self.steering_joints = steering_joints
        self.motor_joints = motor_joints
        
        if HAS_CUSTOM_TRACKER:
            self.tracker = LucasKanadeTracker()
        else:
            self.tracker = None
            
        self.prev_gray = None
        
        # State management
        self.backtrack_counter = 0
        self.stuck_frames = 0

    def get_stabilized_camera(self):
        """
        Extracts a stabilized camera frame from PyBullet.
        Strictly formats the array to prevent cv2.cvtColor errors.
        """
        pos, _ = p.getBasePositionAndOrientation(self.car_id)
        
        eye = [pos[0], pos[1], pos[2] + 0.5]
        target = [pos[0] + 5.0, pos[1], pos[2] + 0.3]
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye,
            cameraTargetPosition=target,
            cameraUpVector=[0, 0, 1]
        )
        proj_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 50)
        
        # Safely retrieve and format PyBullet image data
        cam_data = p.getCameraImage(128, 128, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        # cam_data[2] is the rgbPixels. Force it to be a 128x128x4 uint8 array
        rgb_pixels = np.array(cam_data[2], dtype=np.uint8).reshape((128, 128, 4))
        
        # Slice out the Alpha channel and convert to Grayscale
        gray = cv2.cvtColor(rgb_pixels[:, :, :3], cv2.COLOR_RGB2GRAY)
        return gray

    def compute_vpf_steering(self, current_pos, current_ori, flow_data):
        """
        Calculates Attractive + Repulsive forces safely.
        """
        # --- 1. ATTRACTIVE FORCE ---
        vec_to_goal = GOAL_POS - current_pos[:2]
        global_goal_angle = math.atan2(vec_to_goal[1], vec_to_goal[0])
        
        _, _, yaw = p.getEulerFromQuaternion(current_ori)
        heading_error = global_goal_angle - yaw
        
        # Normalize to [-pi, pi]
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
        f_att_steer = K_ATTRACTIVE * heading_error

        # --- 2. REPULSIVE FORCE ---
        f_rep_steer = 0.0
        
        # Safely parse flow_data, preventing crashes if the tracker returns None or malformed data
        if flow_data is not None:
            try:
                pts, vecs = flow_data
                
                # Verify pts and vecs are valid lists/arrays and have elements
                if pts is not None and vecs is not None and len(pts) > 0 and len(pts) == len(vecs):
                    left_flow = 0.0
                    right_flow = 0.0
                    
                    for i in range(len(pts)):
                        # Handle different shape outputs from custom trackers safely
                        x_img = float(pts[i][0]) if isinstance(pts[i], (list, tuple, np.ndarray)) else float(pts[i])
                        mag = float(np.linalg.norm(vecs[i]))
                        
                        if x_img < 64: 
                            left_flow += mag
                        else:          
                            right_flow += mag
                    
                    f_rep_steer = K_REPULSIVE * (right_flow - left_flow)
            except Exception as e:
                # If custom tracker outputs weird data format, fail gracefully rather than crashing
                pass 

        return f_att_steer + f_rep_steer

    def run_simulation(self):
        print(f"Goal set to {GOAL_POS}. Starting VPF controller...")
        dt = 1./60.
        
        while True:
            # 1. Perception
            curr_gray = self.get_stabilized_camera()
            flow_data = None
            
            if self.prev_gray is not None and self.tracker is not None:
                # We wrap this in a try-except so internal tracker math errors don't crash the bot
                try:
                    flow_data = self.tracker.compute_flow(self.prev_gray, curr_gray)
                except Exception as e:
                    print(f"Tracker internal error: {e}")
                    
            self.prev_gray = curr_gray

            # 2. State Logic
            pos, ori = p.getBasePositionAndOrientation(self.car_id)
            vel, _ = p.getBaseVelocity(self.car_id)
            speed = np.linalg.norm(vel)
            
            # Goal Check
            if np.linalg.norm(np.array(pos[:2]) - GOAL_POS) < 1.0:
                print("GOAL REACHED! Successfully arrived at 31.66, 0")
                break

            # 3. Control & Backtracking
            if self.backtrack_counter > 0:
                target_vel = -5.0
                steer_val = 0.5 
                self.backtrack_counter -= 1
            else:
                if speed < STUCK_VEL:
                    self.stuck_frames += 1
                else:
                    self.stuck_frames = 0
                
                # If stuck against a wall for half a second, backtrack
                if self.stuck_frames > 30: 
                    print("Hit an obstacle! Backtracking...")
                    self.backtrack_counter = BACKTRACK_DUR
                    continue

                raw_steer = self.compute_vpf_steering(np.array(pos), ori, flow_data)
                
                # Cast to standard float and clip to physical joint limits
                steer_val = float(np.clip(raw_steer, -0.6, 0.6))
                target_vel = float(MAX_SPEED)

            # 4. Actuation
            for j in self.motor_joints:
                p.setJointMotorControl2(self.car_id, j, p.VELOCITY_CONTROL, 
                                        targetVelocity=target_vel, force=500)
            for j in self.steering_joints:
                p.setJointMotorControl2(self.car_id, j, p.POSITION_CONTROL, 
                                        targetPosition=steer_val)

            p.stepSimulation()
            time.sleep(dt)

if __name__ == "__main__":
    car_id, steer_joints, motor_joints = setup_simulation(gui=True)
    controller = FlowController(car_id, steer_joints, motor_joints)
    
    try:
        controller.run_simulation()
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
        p.disconnect()