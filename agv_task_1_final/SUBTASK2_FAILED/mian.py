# main.py
import time
import pybullet as p

from simulation_setup import setup_simulation
from flow1 import FlowController


def main():
    car_id, steering_joints, motor_joints = setup_simulation(
        dt=1.0 / 60.0,
        settle_frames=60,
        gui=True,
    )

    controller = FlowController(
        car_id=car_id,
        steering_joints=steering_joints,
        motor_joints=motor_joints,
        image_size=(640, 480),
        gui=True,
    )

    dt = 1.0 / 60.0

    try:
        while True:
            steer, throttle, debug = controller.step()
            controller.apply(steer, throttle)
            p.stepSimulation()
            time.sleep(dt)

            if controller.frame_idx % 30 == 0:
                print(
                    f"frame={controller.frame_idx:04d} "
                    f"steer={steer:+.3f} throttle={throttle:+.3f} "
                    f"tracks={debug['track_count']} risk={debug['risk']:.3f} "
                    f"lat_bias={debug['lateral_bias']:+.3f} "
                    f"foe_bias={debug['foe_bias']:+.3f} "
                    f"speed={debug['speed']:.2f} v_ref={debug['v_ref']:.2f}"
                )

            pos, _ = p.getBasePositionAndOrientation(car_id)
            if pos[0] > 31.0:
                print("[Main] reached the end.")
                break

    except KeyboardInterrupt:
        print("\n[Main] interrupted by user")
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()