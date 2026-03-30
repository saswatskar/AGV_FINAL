import cv2
import numpy as np
import tempfile
import pathlib
import pybullet as p
import pybullet_data
import time
import math
from lucas_kanade import LucasKanadeTracker_1


def make_obstacle_texture(size: int = 128, tile: int = 10) -> str:
    """
    Creates a yellow (#FFD700) / black checkerboard PNG and returns its path.
    Parameters:
    size : int: Image size in pixels (square).     
    tile : int: Checkerboard tile size in pixels.       

    Returns:
    str : Absolute path to the saved PNG texture file.
    """
    img    = np.zeros((size, size, 3), dtype=np.uint8)
    black  = np.array([0,   0,   0],   dtype=np.uint8)  # BGR black
    yellow = np.array([0,   215, 255], dtype=np.uint8)  # BGR yellow (#FFD700)

    for row in range(size):
        for col in range(size):
            img[row, col] = black if (row // tile + col // tile) % 2 == 0 else yellow

    tex_path = pathlib.Path(tempfile.gettempdir()) / "obs_texture.png"
    ok = cv2.imwrite(str(tex_path), img)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed writing texture to: {tex_path}")

    print(f"[Texture] Saved yellow/black checkerboard → {tex_path}")
    return str(tex_path)


# =============================================================================
# ROAD + OBSTACLES
# =============================================================================

def create_road_and_obstacles():
    """
    Builds the full scene in the currently connected PyBullet world:

      - Road surface     : dark grey box, 33.3 m long × 2.32 m wide
      - Lane markings    : dense white dashes at y = 0, ±0.85 m
      - Slalom obstacles : 5 yellow/black textured boxes in alternating lateral positions
      - End wall         : blue box at x ≈ 31.7 m

    Road half-width is 1.16 m on each side of centre (local frame).
    Obstacle positions (x = 6, 12, 18, 24 m) alternate y = +0.38 / −0.38 m.

    Must be called after p.connect() and p.loadURDF("plane.urdf").
    """
    tex_path = make_obstacle_texture(size=128, tile=10)
    tex_id   = p.loadTexture(tex_path)

    # ── Road surface ──────────────────────────────────────────────────────
    rv = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[16.66, 1.16, 0.01],
        rgbaColor=[0.15, 0.15, 0.15, 1]
    )
    rc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[16.66, 1.16, 0.01])
    p.createMultiBody(0, rc, rv, [16.66, 0, 0.01])

    # ── Lane markings — dense white dashes for background optical flow ────
    for x in np.arange(0, 34, 0.4):
        lv = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.12, 0.03, 0.01],
            rgbaColor=[1, 1, 1, 1]
        )
        #p.createMultiBody(0, -1, lv, [x,  0.0,  0.02])
        p.createMultiBody(0, -1, lv, [x,  0.85, 0.02])
        p.createMultiBody(0, -1, lv, [x, -0.85, 0.02])

    # ── Slalom obstacles — white base so yellow/black texture shows fully ─
    obs_extents = [0.25, 0.45, 0.35]
    for i, x in enumerate(range(6, 30, 6)):
        y      = 0.38 if i % 2 == 0 else -0.38
        ov     = p.createVisualShape(
            p.GEOM_BOX, halfExtents=obs_extents,
            rgbaColor=[1, 1, 1, 1]
        )
        oc     = p.createCollisionShape(p.GEOM_BOX, halfExtents=obs_extents)
        obs_id = p.createMultiBody(10, oc, ov, [x, y, 0.35])
        p.changeVisualShape(obs_id, -1, textureUniqueId=tex_id)
        print(f"[Obstacle {i}] x={x}m  y={y:+.2f}m  texture applied")

    # ── End wall (blue — visually distinct from obstacles) ────────────────
    wv = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.5, 1.16, 0.5],
        rgbaColor=[0.1, 0.3, 0.9, 1]
    )
    wc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 1.16, 0.5])
    p.createMultiBody(0, wc, wv, [31.66, 0, 0.5])
    print("[End wall] placed at x=31.66 m")


# =============================================================================
# CAR
# =============================================================================

def create_car(start_pos=None, start_orn=None, global_scaling=1.8):
    """
    Loads the PyBullet racecar URDF and returns its ID along with
    categorised joint lists

    Parameters:
    start_pos : list[float], optional: [x, y, z] spawn position. Defaults to [0, 0, 0.25].
    start_orn : list[float], 
    global_scaling : float: URDF scaling factor (1.8 matches the road width).

    Returns:
    car_id : int: PyBullet body ID of the car.
    steering_joints : list[int]: Joint indices whose names contain 'steer'.
    motor_joints : list[int]: Joint indices whose names contain 'wheel'.
    """
    if start_pos is None:
        start_pos = [0.5, 0, 0.25]
    if start_orn is None:
        start_orn = p.getQuaternionFromEuler([0, 0, 0])

    car_id = p.loadURDF(
        "racecar/racecar.urdf",
        start_pos, start_orn,
        globalScaling=global_scaling
    )
    p.changeDynamics(car_id, -1, ccdSweptSphereRadius=0.1)

    steering_joints, motor_joints = [], []
    for i in range(p.getNumJoints(car_id)):
        name = p.getJointInfo(car_id, i)[1].decode('utf-8')
        if 'steer' in name.lower():
            steering_joints.append(i)
        elif 'wheel' in name.lower():
            motor_joints.append(i)

    print(f"[Car] body_id={car_id}  "
          f"steering_joints={steering_joints}  "
          f"motor_joints={motor_joints}")
    return car_id, steering_joints, motor_joints


# =============================================================================
# CONVENIENCE WRAPPER
# =============================================================================

def setup_simulation(dt=1.0 / 60.0, settle_frames=60, gui=True):
    """
    Full simulation initialisation in one call.

    Steps
    -----
    1. Connect to PyBullet (GUI or DIRECT).
    2. Set gravity and load the ground plane.
    3. Build road, lane markings, obstacles, and end wall.
    4. Spawn the racecar and settle its suspension.

    Parameters
    ----------
    dt : float: Physics timestep in seconds. Default 1/60 s.
    settle_frames : int: Number of physics steps to run before returning, so the car
        suspension reaches equilibrium. Default 60.
    gui : bool : If True, opens the PyBullet GUI window. If False, runs headless
        (DIRECT mode) — useful for unit tests or batch runs.

    Returns
    -------
    car_id : int
    steering_joints : list[int]
    motor_joints : list[int]
    """
    mode = p.GUI if gui else p.DIRECT
    p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(dt)
    p.loadURDF("plane.urdf")

    create_road_and_obstacles()
    car_id, steering_joints, motor_joints = create_car()

    print(f"[Setup] Settling suspension for {settle_frames} frames …")
    for _ in range(settle_frames):
        p.stepSimulation()
        time.sleep(dt)
    print("[Setup] Ready.")

    return car_id, steering_joints, motor_joints

def steer_towards_force(car_id, steering_joints, motor_joints, force):
    fx, fy = force

    # desired direction
    desired_yaw = np.arctan2(fy, fx)

    # current yaw
    _, orn = p.getBasePositionAndOrientation(car_id)
    _, _, current_yaw = p.getEulerFromQuaternion(orn)

    # error
    yaw_error = desired_yaw - current_yaw
    yaw_error = (yaw_error + np.pi) % (2*np.pi) - np.pi

    # steering
    steer = np.clip(0.8 * yaw_error, -0.5, 0.5)

    for j in steering_joints:
        p.setJointMotorControl2(
            car_id, j,
            p.POSITION_CONTROL,
            targetPosition=steer,
            force=50
        )

    # speed proportional to force
    speed = np.linalg.norm([fx, fy])

    for j in motor_joints:
        p.setJointMotorControl2(
            car_id, j,
            p.VELOCITY_CONTROL,
            targetVelocity=5 * speed,
            force=800
        )

debug_text_id = None

def show_force_debug(pos, force, debug_text_id):
    text = f"pos={np.round(pos,2)}  F={np.round(force,2)}"

    if debug_text_id is None:
        debug_text_id = p.addUserDebugText(
            text,
            [0, 0, 1.5],   # position in world
            textSize=1.5,
            lifeTime=0     # persistent
        )
    else:
        p.addUserDebugText(
            text,
            [0, 0, 1.5],
            textSize=0.75,
            replaceItemUniqueId=debug_text_id
        )

    return debug_text_id

def attractive_force() :
    return [50,0]

def radial_repulsive_force(pos,
                          center=np.array([6.0, -0.4]),
                          k=500.0,
                          p=5.0,
                          epsilon=1e-3,
                          max_force=200.0):
    """
    Radially outward repulsive force field.

    Parameters:
    pos : array-like [x, y]
    center : array-like [cx, cy]
    k : float → strength of field
    p : float → decay exponent
    epsilon : float → avoids division by zero
    max_force : float → clamp for stability

    Returns:
    [fx, fy]
    """

    pos = np.array(pos)
    d = pos - center

    r = np.linalg.norm(d)

    # unit direction
    if r < 1e-8:
        direction = np.array([0.0, 0.0])
    else:
        direction = d / r

    # magnitude (blows up near center)
    magnitude = k / ((r + epsilon) ** p)

    # clamp to avoid insane values
    magnitude = min(magnitude, max_force)

    force = magnitude * direction
    a=force.tolist()
    a[1] = -a[1]
    if pos[0] > center[0]-0.5 :
        a = [0,0]
    return a

def wall_Force(y) :
    if y > 0.5 : 
        return [0,-20000]
    elif y<-0.5 :
        return [0,20000]
    else : 
        return [0 , -y*50]
    

def bring_to_normal(f) :

    mag = math.sqrt(f[0]*f[0]+f[1]*f[1])

    return [f[0]/mag , f[1]/mag]

#i will make controler





    # def calc_repulsive_force() :



    # def calculate_stur() :
# =============================================================================
# STANDALONE DEMO
# =============================================================================

if __name__ == "__main__":
    car_id, steer_j, motor_j = setup_simulation()
    print("\nSimulation running. Close the PyBullet window or press Ctrl+C to exit.")
    dt = 1.0 / 60.0
    try:
        while True:
            # Spin the wheels at a gentle speed so you can see the car move
            pos, orn = p.getBasePositionAndOrientation(car_id)
            x , y , z = pos
            a = attractive_force()
            r1 = radial_repulsive_force([x,y] , center=np.array([6,-0.4]))
            r2 = radial_repulsive_force([x,y] , center=np.array([12,0.4]))
            r3 = radial_repulsive_force([x,y] , center=np.array([18,-0.4]))
            r4 = radial_repulsive_force([x,y] , center=np.array([24,0.4]))


            r = []
            for i in range(0,2) :
                r.append(r1[i] + r2[i]+r3[i]+r4[i])
            w = wall_Force(y)
            total_force = []
            for i in range(0,2) :
                total_force.append(a[i]+r[i]+w[i])

            steer_towards_force(car_id , steer_j , motor_j ,bring_to_normal(total_force) )
            debug_text_id = show_force_debug([x,y],r, debug_text_id)
            
            p.stepSimulation()
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            p.disconnect()
        except Exception:
            pass
        print("Simulation ended.")