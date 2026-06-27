# This example demonstrates teleoperation of the Sphero robot using keyboard inputs,
# with an optional simulator-only mode for testing without a physical robot.
# It also shows how to log data from both the simulator and the real robot for later analysis.

import time
import argparse
import numpy as np
import pygame
from contextlib import ExitStack, contextmanager

from sphero_unsw.sphero_edu import SpheroEduAPI
from sphero_env.robot.connect import scan_and_connect
from sphero_env.robot.robot import Robot
from sphero_env.envs import SpheroEnv

# This function creates a new simulator environment
def make_sim_env():
    return SpheroEnv(
        dt=0.1,
        max_steps=5000,
        action_limit=8,
        vel_limit=0.25,
        world_width=5.0,
        world_height=5.0,
        max_turn_rate=6.0,
        speed_scale=4.0,
        goal_pos=(0.5, 0.5),
        goal_tolerance=0.05,
        occupancy_grid=None,
        obs_noise_std_pos=0.05,
        process_noise_std_speed=0.025,
        process_noise_std_heading=0.05,
        obs_noise_std_vel=0,
        render_mode="human",
        window_size=(800, 800),
    )


# This function creates a new Robot environment that interfaces with the physical Sphero robot
def make_real_env(api):
    return Robot(
        api=api,
        dt=0.1,
        max_steps=5000,
        vel_limit=1,
        world_width=1.0,
        world_height=1.0,
        goal_pos=(0.8, 0.8),
        goal_tolerance=0.15,
    )


@contextmanager
def managed_sim_env():
    env = make_sim_env()
    # Set up logging for the simulator environment
    env.set_log_path("logs/sim_teleop_log.csv")
    env.reset()
    env.start_logging()
    try:
        yield env
    finally:
        env.stop_logging()
        env.close()


@contextmanager
def managed_robot_env():
    # Scan for and connect to a Sphero robot, then set up logging for the robot environment
    selected_toy, _ = scan_and_connect()
    print(f"Selected: {selected_toy.name}")

    with SpheroEduAPI(selected_toy) as api:
        robot_env = make_real_env(api)
        robot_env.set_log_path("logs/sphero_teleop_log.csv")
        robot_env.reset()
        robot_env.start_logging()
        try:
            yield robot_env
        finally:
            robot_env.stop_logging()
            robot_env.emergency_stop()
            robot_env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Teleoperate Sphero with optional simulator-only mode")
    parser.add_argument("--sim", action="store_true", help="Run simulator-only mode (no robot connection)")
    return parser.parse_args()


def main(sim_only=False):
    """Main function for teleoperation of Sphero robot with simulator."""
    with ExitStack() as stack:
        env = stack.enter_context(managed_sim_env())
        robot_env = stack.enter_context(managed_robot_env()) if not sim_only else None

        env.render()

        stack.callback(pygame.quit)
        stack.callback(lambda: print("Stopped. Teleop closed."))

        mode_text = "Simulator only" if sim_only else "Robot + Simulator"
        print(f"\nTele-op ready ({mode_text}):")
        print("  W       = move forward")
        print("  A/D     = turn left/right")
        print("  S       = move backward")
        print("  SPACE   = emergency stop")
        print("  +/-     = speed")
        print("  Q       = quit\n")

        speed_norm = 0.25  # normalized speed [0, 1]
        current_heading = 0.0 if robot_env is None else float(robot_env.get_odom_state()[2])

        running = True
        moving = False
        last_speed_change_time = 0.0

        while running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False

                    elif event.key == pygame.K_SPACE:
                        if robot_env is not None:
                            robot_env.emergency_stop()
                        moving = False
                        print("Emergency stop")

                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        now = time.time()
                        if now - last_speed_change_time > 0.08:
                            speed_norm = min(1.0, speed_norm + 0.1)
                            print(f"Speed: {speed_norm:.1f}")
                            last_speed_change_time = now

                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        now = time.time()
                        if now - last_speed_change_time > 0.08:
                            speed_norm = max(0.0, speed_norm - 0.1)
                            print(f"Speed: {speed_norm:.1f}")
                            last_speed_change_time = now

            # Handle continuous key presses
            pressed = pygame.key.get_pressed()

            v_cmd = 0.0
            if pressed[pygame.K_w]:
                v_cmd = speed_norm * 1.0  # forward
                moving = True
            elif pressed[pygame.K_s]:
                v_cmd = -speed_norm * 1.0  # backward
                moving = True
            else:
                moving = False

            # Handle turning
            turn_rate = 0.2  # radians per frame
            if pressed[pygame.K_a]:
                current_heading -= turn_rate  # turn left

            if pressed[pygame.K_d]:
                current_heading += turn_rate  # turn right

            # Normalize heading to [-pi, pi]
            current_heading = (current_heading + np.pi) % (2 * np.pi) - np.pi

            # Create action [v, theta]
            action = np.array([v_cmd, current_heading], dtype=np.float32)

            # Control robot directly (if connected)
            if robot_env is not None:
                obs, _, terminated, truncated, info = robot_env.step(action)
                # print(f"Robot state: {info['state_odom']}, Collision: {info['collision']}, Acceleration: {info['acceleration']}, Orientation: {info['orientation']}, Gyro: {info['gyroscope']}, Velocity: {info['velocity']}")

            # Step simulator directly and update its logging/visualization state.
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()

            if robot_env is not None and not moving:
                robot_env.emergency_stop()

            # Update overlay trajectories
            real_traj = None
            if robot_env is not None:
                real_x, real_y, _ = robot_env.get_live_traj()
                real_traj = np.column_stack([np.array(real_x), np.array(real_y)]) if len(real_x) > 0 else None
            else:
                # In sim-only mode, use simulator ground-truth path as the real trajectory overlay.
                sim_true_x, sim_true_y, _ = env.get_live_traj()
                real_traj = (
                    np.column_stack([np.array(sim_true_x), np.array(sim_true_y)])
                    if len(sim_true_x) > 0
                    else None
                )
            sim_x, sim_y, _ = env.get_live_traj()
            odom_traj = np.column_stack([np.array(sim_x), np.array(sim_y)]) if len(sim_x) > 0 else None
            env.set_overlay_trajectories(
                real_traj=real_traj,
                odom_traj=odom_traj,
                true_traj=None,
            )

            env.render()

            time.sleep(0.01)  # Small delay to prevent busy-waiting


if __name__ == "__main__":
    args = parse_args()
    main(sim_only=args.sim)
