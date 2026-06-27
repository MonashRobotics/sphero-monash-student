"""
point_env

A simple kinematic point-mass Gymnasium environment for teaching
intelligent robotics concepts.

Usage:
    import gymnasium as gym
    import point_env

    env = gym.make("SpheroEnv-v0")
"""

from sphero_env.envs.sphero_env import SpheroEnv
from sphero_env.envs.visualiser import Visualiser, LabVisualiser
from sphero_env.envs.maze_generator import grid_to_world, MazeGenerator

# Optional: provide a helper registration function
import gymnasium as gym

def register_env():
    """Register the SpheroEnv environment with gymnasium."""
    gym.register(
        id="SpheroEnv-v0",
        entry_point="sphero_env.envs.sphero_env:SpheroEnv",
    )

# Automatically register on import if you like:
register_env()

__all__ = ["SpheroEnv", "Visualiser", "LabVisualiser", "MazeGenerator", "grid_to_world", "register_env"]
