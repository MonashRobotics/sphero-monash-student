# Import necessary libraries
from sphero_env.robot.connect import scan_and_connect
from sphero_unsw.sphero_edu import SpheroEduAPI
from sphero_env.robot.robot import Robot
from sphero_env.envs import SpheroEnv

import argparse
import numpy as np

from contextlib import ExitStack, contextmanager

LAB1_SEED = 0
MAX_STEPS = 500

### Custom dynamics function for the Sphero robot - replace this with the one you developed in Lab 1
def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi  # Normalize to [-pi, pi)

def dynamics(state, action):
        """
        Compute the next state given current state and action using the base dynamics without noise.
        This can be rewritten to improve the model.
        """
        x, y, heading, speed = state
        speed_cmd, turn_rate_cmd = action
        # Simple unicycle model dynamics
        heading_new = wrap_angle(heading + turn_rate_cmd * 0.1)
        speed_new = np.clip(speed + speed_cmd * 0.1, 0, 1.0)
        x_new = x + speed_new * np.sin(heading_new) * 0.1
        y_new = y + speed_new * np.cos(heading_new) * 0.1
        return np.array([x_new, y_new, heading_new, speed_new], dtype=np.float32)

### If needed, add the EKF from lab 2 here too, and integrate below.

### Implement a planner and controller for the Sphero robot to navigate to a goal position in the environment.
class Planner:
    def __init__(self, map, dt=0.1):
        self.map = map
        self.dt = dt

    def plan(self, state, goal):
        """
        Fill in this function to implement a simple planner that computes the action based on the current state, 
        the map and the goal position. The planner should return a sequence of 2D waypoints to reach the goal.
        """

        waypoints = [np.array([0.5, 0.5])]  # Replace this with your planner's output

        return waypoints  # Replace this with your planner's output
    
class Controller:
    def __init__(self, dt=0.1):
        self.dt = dt

    def compute_action(self, state, waypoint):
        """
        Fill in this function to implement a simple controller that computes the action based on the current state and the waypoint.
        """

        action = np.array([0.0, 0.0])  # Replace this with your controller's output

        return action  # Replace this with your controller's output

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
def managed_env(sim: bool):
    if sim:
        sim_env = make_sim_env()
        sim_env.set_log_path("logs/lab3_sim.csv")
        sim_env.start_logging()
        try:
            yield sim_env
        finally:
            sim_env.stop_logging()
            sim_env.close()
    else:
        with ExitStack() as stack:
            selected_toy, _ = scan_and_connect()
            print(f"Selected: {selected_toy.name}")

            api = stack.enter_context(SpheroEduAPI(selected_toy))
            real_env = make_real_env(api)
            real_env.set_log_path("logs/lab3_real.csv")

            real_env.start_logging()
            try:
                yield real_env
            finally:
                real_env.close()
                real_env.stop_logging()

def control_loop(control_env):

    obs, _ = control_env.reset(seed=LAB1_SEED)
    rng = np.random.default_rng(LAB1_SEED)

    controller = Controller(dt=control_env.dt)
    planner = Planner(map=control_env.occupancy_grid, dt=control_env.dt)

    waypoints = planner.plan(obs, control_env.goal_pos)

    steps = 0
    while steps < MAX_STEPS:

        for waypoint in waypoints:
            action = controller.compute_action(obs, waypoint)
            obs, _, terminated, truncated, info = control_env.step(action)

            if (obs[0]-waypoint[0])**2 + (obs[1]-waypoint[1])**2 < control_env.goal_tolerance**2:
                continue  # Move to the next waypoint if close enough

            if (obs[0]-control_env.goal_pos[0])**2 + (obs[1]-control_env.goal_pos[1])**2 < control_env.goal_tolerance**2:
                break  # Move to the next waypoint if close enough
            
            control_env.render()
        
        steps += 1

    control_env.emergency_stop()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="Run simulation")
    args = parser.parse_args(argv)

    with managed_env(args.sim) as control_env:
        control_loop(control_env)

if __name__ == "__main__":
    main()