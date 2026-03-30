import cv2
import numpy as np
import pybullet as p
import time
from simulation_setup import setup_simulation

# --- TUNABLE CONTROLLER PARAMETERS ---
ALPHA = 0.8              # Attractive force gain (Pull to goal)
GAMMA_LATERAL = 8000.0   # Repulsive lateral force gain (Steering away)
GAMMA_LONG = 0.5         # Repulsive longitudinal gain (Braking on low TTC)
SIGMA = 51               # Gaussian smoothing kernel size for the obstacle field
MAX_SPEED = 15.0         # Maximum target speed for the motor
STEER_GAIN = 0.005       # Maps lateral force to PyBullet steering joint angles

# --- CAMERA PARAMETERS ---
IMG_W, IMG_H = 320, 240

def get_camera_frame(car_id):
    """Captures RGB and Grayscale frames from the car's perspective."""
    pos, orn = p.getBasePositionAndOrientation(car_id)
    yaw = p.getEulerFromQuaternion(orn)[2]
    
    # Mount camera slightly above and forward of the car's center
    cam_pos = [pos[0] + 0.6 * np.cos(yaw), pos[1] + 0.6 * np.sin(yaw), pos[2] + 0.5]
    target_pos = [cam_pos[0] + 5.0 * np.cos(yaw), cam_pos[1] + 5.0 * np.sin(yaw), cam_pos[2] - 0.2]
    
    view_matrix = p.computeViewMatrix(cam_pos, target_pos, [0, 0, 1])
    proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=IMG_W/IMG_H, nearVal=0.1, farVal=100.0)
    
    img_arr = p.getCameraImage(IMG_W, IMG_H, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb = np.reshape(img_arr[2], (IMG_H, IMG_W, 4))[:, :, :3]
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    return gray, rgb

def compute_foe(pts_old, flows):
    """Computes Focus of Expansion using Least Squares (Eq 1 & 2)."""
    vx = flows[:, 0]
    vy = flows[:, 1]
    x = pts_old[:, 0]
    y = pts_old[:, 1]
    
    A = np.column_stack((vy, -vx))
    b = x * vy - y * vx
    
    # Solve (A^T A)^-1 A^T b
    foe, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    if len(foe) == 2:
        return foe
    return np.array([IMG_W/2, IMG_H/2])

def run_controller():
    car_id, steer_j, motor_j = setup_simulation(gui=True)
    prev_gray, _ = get_camera_frame(car_id)
    
    # Setup feature tracker params
    lk_params = dict(winSize=(25, 25), maxLevel=3, 
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=200, qualityLevel=0.03, minDistance=7, blockSize=7)
    
    while True:
        curr_gray, curr_rgb = get_camera_frame(car_id)
        
        # 1. Optical Flow Extraction
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        if p0 is not None:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            flows = good_new - good_old
            
            if len(flows) > 5:
                # 2. Compute FOE
                foe = compute_foe(good_old, flows)
                
                # 3. Obstacle Plane O_t(x,y) via Otsu on flow magnitude
                mags = np.linalg.norm(flows, axis=1)
                mags_norm = np.uint8(255 * (mags / (np.max(mags) + 1e-5)))
                _, thresh = cv2.threshold(mags_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                O_t = np.zeros_like(curr_gray, dtype=np.float32)
                ttc_sum, obs_count = 0, 0
                
                for i, (pt, mag, val) in enumerate(zip(good_new, mags, mags_norm)):
                    if val >= thresh:  # Threshold flags it as an obstacle
                        x, y = int(pt[0]), int(pt[1])
                        cv2.circle(O_t, (x, y), 10, 1.0, -1)
                        
                        # TTC calculation (Eq 3)
                        dist_foe = np.linalg.norm(pt - foe)
                        ttc = dist_foe / (mag + 1e-5)
                        ttc_sum += ttc
                        obs_count += 1

                # 4. Gaussian Smoothing & Gradient (Eq 4, 5, 6)
                G_O_t = cv2.GaussianBlur(O_t, (SIGMA, SIGMA), 0)
                g_x = cv2.Sobel(G_O_t, cv2.CV_64F, 1, 0, ksize=5)
                
                # 5. Potential Field Forces (Eq 7, 8, 9)
                # Attractive Force (Target = center horizon of image)
                target_x, target_y = IMG_W / 2, IMG_H / 2
                ego_x, ego_y = IMG_W / 2, IMG_H
                
                # Vector to target
                F_att_x = ALPHA * (target_x - ego_x)
                F_att_y = ALPHA * np.sqrt((target_x - ego_x)**2 + (target_y - ego_y)**2)
                
                # Repulsive Force
                # We weight g_x by its distance from the vehicle center to push the car correctly
                x_indices = np.arange(IMG_W) - (IMG_W / 2)
                weighted_g_x = g_x * np.sign(x_indices)
                
                F_rep_x = GAMMA_LATERAL * np.sum(weighted_g_x) / (IMG_W * IMG_H)
                F_rep_y = GAMMA_LONG * (1.0 / (ttc_sum / max(obs_count, 1) + 1e-5)) if obs_count > 0 else 0
                
                # Total Forces
                F_total_x = F_att_x - F_rep_x
                F_total_y = F_att_y - F_rep_y
                
                # 6. Actuation Mapping
                target_steer = np.clip(-F_total_x * STEER_GAIN, -0.5, 0.5)
                target_speed = np.clip(MAX_SPEED * (F_total_y / F_att_y), 2.0, MAX_SPEED)
                
                for j in steer_j:
                    p.setJointMotorControl2(car_id, j, p.POSITION_CONTROL, targetPosition=target_steer)
                for j in motor_j:
                    p.setJointMotorControl2(car_id, j, p.VELOCITY_CONTROL, targetVelocity=target_speed, force=1000)
                
                # Visualization (Optional debug)
                cv2.circle(curr_rgb, (int(foe[0]), int(foe[1])), 5, (0, 0, 255), -1)
                cv2.imshow("Optical Flow & FOE", curr_rgb)
                cv2.imshow("Smoothed Obstacle Field (g_x)", np.abs(g_x) / np.max(np.abs(g_x) + 1e-5))
                cv2.waitKey(1)
                
        prev_gray = curr_gray.copy()
        p.stepSimulation()
        time.sleep(1./60.)

if __name__ == "__main__":
    run_controller()