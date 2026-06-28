import time
import argparse
import numpy as np
import pygame
from contextlib import ExitStack, contextmanager

from sphero_unsw.sphero_edu import SpheroEduAPI
from sphero_env.robot.connect import scan_and_connect
from sphero_env.robot.robot import Robot
from sphero_env.envs import SpheroEnv


### Custom dynamics function for the Sphero robot - replace this with the one you developed in Lab 1
def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi  # Normalize to [-pi, pi)

def dynamics(state, action):
        """
        Compute the next state given current state and action using the base dynamics without noise.
        This is a bad model of the robot.
        """
        x, y, heading, speed = state
        speed_cmd, turn_rate_cmd = action
        # Simple unicycle model dynamics
        heading_new = wrap_angle(heading + turn_rate_cmd * 0.1)
        speed_new = np.clip(speed_cmd, -2.0, 2.0)
        x_new = x + speed_new * np.sin(heading_new) * 0.1
        y_new = y + speed_new * np.cos(heading_new) * 0.1
        return np.array([x_new, y_new, heading_new, speed_new], dtype=np.float32)


### EKF class to track odometry and perform state estimation for the Sphero robot
# Complete this class to implement the EKF algorithm for state estimation based on your dynamics and measurement models.

class EKF:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.state_est = np.zeros(4)  # [x, y, heading, speed]
        self.P = np.eye(4) * 0.1  # Initial covariance
        self.Q = np.diag([0.001, 0.001, 0.001, 0.001])  # Process noise covariance
        self.R = np.diag([0.005, 0.005])  # Measurement noise covariance

    def predict(self, action):
        # Predict the next state using the dynamics function
        self.state_est = dynamics(self.state_est, action)
        
        # Update the covariance matrix using a simple linear approximation of the dynamics
        self.P = self.P # Replace this with a proper Jacobian-based update
        return self.state_est, self.P

    def update(self, measurement):
        # Update the state estimate with a new measurement
        
        self.state_est = self.state_est  # Replace this with a proper Kalman gain update
        self.P = self.P  # Replace this with a proper covariance update

        return self.state_est, self.P
    
### Control loop to handle the action and update the environments, 
# edit this to include the EKF prediction and update steps, and to visualize the belief state in the simulator.
def control_loop(env, ekf, robot_env=None,action=None, moving=False):
    """
    Control loop to handle the action and update the environments.
    This function is called in the main loop to step both the simulator and the robot (if connected).
    """
    # Step simulator directly and update its logging/visualization state.
    sim_obs, _, _, _, sim_info = env.step(action)

    # Control robot directly (if connected)
    if robot_env is not None:
        robot_obs, _, _, _, robot_info = robot_env.step(action)
        # print(f"Robot state: {info['state_odom']}, Collision: {info['collision']}, Acceleration: {info['acceleration']}, Orientation: {info['orientation']}, Gyro: {info['gyroscope']}, Velocity: {info['velocity']}")
 
    if robot_env is not None and not moving:
        robot_env.emergency_stop()

     # Call the EKF to predict and update the state estimate based on the action and observation

    ekf.predict(action)  # Predict the next state using the EKF
    if robot_env is not None:
        ekf.update(robot_obs)  # Update the EKF with the odometry measurement - you may want to also include other measurements if available from the info dictionary (e.g., IMU, gyro, etc.)
    else:
        ekf.update(sim_obs)  # Update the EKF with the simulator observation

    env.vis.set_belief(ekf.state_est, ekf.P)  # Update the simulator with the EKF state estimate to visualise the belief state


    env.render()

# This function creates a new simulator environment
def make_sim_env():
    return SpheroEnv(
        dt=0.1,
        max_steps=5000,
        vel_limit=0.15,
        world_width=5.0,
        world_height=5.0,
        goal_pos=(0.5, 0.5),
        goal_tolerance=0.1,
        occupancy_grid=None,
        dynamics=dynamics,
        obs_noise_std_pos=0.05,
        process_noise_std_speed=0.005,
        process_noise_std_heading=0.01,
        obs_noise_std_vel=0.025,
        render_mode="human",
        window_size=(800, 800),
    )

def make_real_env(api):
    return Robot(
        api=api,
        dt=0.1,
        max_steps=5000,
        vel_limit=0.15,
        world_width=5.0,
        world_height=5.0,
        goal_pos=(0.5, 0.5),
        goal_tolerance=0.1,
        render_mode="human",
        window_size=(800, 800),
    )

@contextmanager
def managed_sim_env():
    env = make_sim_env()
    # Set up logging for the simulator environment
    env.set_log_path("logs/lab2_sim.csv")
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
        robot_env.set_log_path("logs/lab2_robot.csv")
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

        ekf = EKF(dt=0.1)  # Initialize the EKF for state estimation

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

            control_loop(env, ekf=ekf, robot_env=robot_env, action=action, moving=moving)  # Call the control loop to handle the action and update the environments

            if robot_env is not None:
                time.sleep(0.01)  # Small delay to prevent busy-waiting
            else:
                time.sleep(0.1)  # Small delay to prevent busy-waiting in sim-only mode

if __name__ == "__main__":
    args = parse_args()
    main(sim_only=args.sim)
