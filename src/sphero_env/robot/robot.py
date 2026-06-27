import time
import threading
from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from sphero_env.envs.visualiser import Visualiser
from sphero_unsw.sphero_edu import SpheroEduAPI


class Robot(gym.Env):
    """
    Real Sphero robot wrapper that mimics the SpheroEnv interface.
    
    Allows interchangeable use between simulation (SpheroEnv) and real hardware (Robot).
    
    The Robot class:
    - Wraps SpheroEduAPI for hardware control
    - Tracks odometry state [x_o, y_o, vx_o, vy_o]
    - Maintains a mirror of the observation interface
    - Provides reset() and step() methods compatible with SpheroEnv
    
    Note: Real robot doesn't have "true" ground truth, so we track odometry as the best estimate.
    """

    metadata = {"render_modes": [], "render_fps": 60}

    def __init__(
        self,
        api: SpheroEduAPI,
        # Time & dynamics
        dt: float = 0.05,
        max_steps: int = 5000,
        vel_limit: float = 0.5,
        settle_steps: int = 15,

        # World dimensions (meters) - for compatibility
        world_width: float = 2.0,
        world_height: float = 2.0,

        # Goal configuration (meters in world frame)
        goal_pos: Tuple[float, float] = (0.0, 0.0),
        goal_tolerance: float = 0.2,

        # Observation noise (std dev)
        obs_noise_std_pos: float = 0.05,
        obs_noise_std_vel: float = 0.05,

        # Logging
        csv_path: Optional[str] = None,

        # Rendering
        render_mode: Optional[str] = None,
        window_size: Tuple[int, int] = (600, 600),
    ):
        super().__init__()

        self.api = api

        # Logging + visualization settings
        self.last_command = None  # (heading_cmd_rad, speed_cmd_norm, timestamp)

        # Thread safety
        self._lock = threading.Lock()
        self.dt = dt
        self.max_steps = max_steps
        self.vel_limit = vel_limit
        self.settle_steps = int(max(0, settle_steps))

        # World geometry
        self.world_width = world_width
        self.world_height = world_height
        self.x_min = -world_width / 2.0
        self.x_max = world_width / 2.0
        self.y_min = -world_height / 2.0
        self.y_max = world_height / 2.0

        # Noise parameters
        self.obs_noise_std_pos = obs_noise_std_pos
        self.obs_noise_std_vel = obs_noise_std_vel

        # Internal states (tracking real robot via odometry)
        self.state_odom: Optional[np.ndarray] = None   # [x_o, y_o, heading_o, speed_o]
        self.state_true: Optional[np.ndarray] = None   # same as odom for real robot
        self.last_obs: Optional[np.ndarray] = None
        self.step_count = 0
        self.latest_est_mean: Optional[np.ndarray] = None
        self.latest_est_cov: Optional[np.ndarray] = None

        # Software collision sensing state. This feeds obs[4] and info["collision"].
        self.last_collision_flag: float = 0.0
        self._last_accel_mag: Optional[float] = None
        self._last_speed_cm_s: Optional[float] = None
        self._last_gyro_mag: Optional[float] = None
        self._last_collision_time: float = 0.0
        self._last_collision_debug: Dict[str, float] = {}

        # Collision thresholds. Tune these after printing collision_debug values
        # while driving into a wall/object a few times.
        self.collision_dead_time: float = 0.25
        self.collision_accel_threshold: float = 1.5
        self.collision_jerk_threshold: float = 0.8
        self.collision_speed_drop_threshold: float = 10.0
        self.stationary_collision_jerk_threshold: float = 1.2
        self.collision_gyro_spike_threshold: float = 100.0
        self.collision_gyro_mag_threshold: float = 200.0

        # Goal
        self.goal_pos = np.array(goal_pos, dtype=np.float32)
        self.goal_tolerance = goal_tolerance
        self.current_setpoint = self.goal_pos.copy()
        self._use_custom_setpoint = False

        # Gym spaces
        obs_low = np.array(
            [self.x_min, self.y_min, -np.pi, -self.vel_limit, 0.0],
            dtype=np.float32,
        )
        obs_high = np.array(
            [self.x_max, self.y_max, np.pi, self.vel_limit, 1.0],
            dtype=np.float32,
        )
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-self.vel_limit, -np.pi], dtype=np.float32),
            high=np.array([self.vel_limit, np.pi], dtype=np.float32),
            dtype=np.float32,
        )

        self.vis = Visualiser(
            world_width=self.world_width,
            world_height=self.world_height,
            goal_pos=self.goal_pos,
            render_mode=render_mode,
            window_size=window_size,
            occupancy_grid=None,
            grid_resolution=0.1,
            csv_path=csv_path,
        )

    def start_logging(self):
        self.vis.start_logging()

    def stop_logging(self):
        self.vis.stop_logging()

    def set_log_path(self, csv_path: Optional[str]):
        """Set/override CSV log path for the internal visualiser."""
        self.vis.csv_path = csv_path

    def set_current_setpoint(self, setpoint_xy: Optional[np.ndarray]):
        """Set the current position setpoint used for logging."""
        if setpoint_xy is None:
            self._use_custom_setpoint = False
            self.current_setpoint = self.goal_pos.copy()
            return
        setpoint_arr = np.asarray(setpoint_xy, dtype=np.float32).reshape(-1)
        if setpoint_arr.shape[0] < 2:
            raise ValueError("setpoint_xy must contain at least 2 values [x, y].")
        self.current_setpoint = setpoint_arr[:2].copy()
        self._use_custom_setpoint = True

    def update_estimate(
        self,
        est_mean: Optional[np.ndarray] = None,
        est_cov: Optional[np.ndarray] = None,
    ):
        """Update estimator belief used for rendering and step-wise logging."""
        if est_mean is None or est_cov is None:
            return
        self.vis.set_belief(est_mean, est_cov)
        self.latest_est_mean = np.asarray(est_mean, dtype=np.float32).copy()
        self.latest_est_cov = np.asarray(est_cov, dtype=np.float32).copy()

    def update_visualization(
        self,
        action: Optional[np.ndarray] = None,
        est_mean: Optional[np.ndarray] = None,
        est_cov: Optional[np.ndarray] = None,
    ):
        """Backward-compatible wrapper; logging now happens in step()."""
        _ = action
        self.update_estimate(est_mean=est_mean, est_cov=est_cov)

    def get_live_traj(self):
        """Get current trajectory data.

        Returns:
            traj_x, traj_y, None.
        """
        return self.vis.get_traj()

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _heading_deg_to_rad(heading_deg: float) -> float:
        """Convert hardware heading in degrees to wrapped radians."""
        return Robot._wrap_angle(np.deg2rad(float(heading_deg)))

    # ---- Core API (matching SpheroEnv interface) ---- #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """
        Reset the robot to initial state.
        
        For real hardware, this initializes odometry tracking to origin.
        """
        super().reset(seed=seed)

        # Reset to origin with zero velocity
        x0 = 0.0
        y0 = 0.0
        heading0 = 0.0
        speed0 = 0.0

        self.last_collision_flag = 0.0
        self._last_accel_mag = None
        self._last_speed_cm_s = None
        self._last_gyro_mag = None
        self._last_collision_time = 0.0
        self._last_collision_debug = {}

        # Initialize state
        self.state_odom = np.array([x0, y0, heading0, speed0], dtype=np.float32)
        self.state_true = self.state_odom.copy()
        self.step_count = 0
        self.current_setpoint = self.goal_pos.copy()
        self._use_custom_setpoint = False
        self.latest_est_mean = None
        self.latest_est_cov = None

        self.vis.reset()
        self.vis.set_goal(self.goal_pos)

        # Stop robot
        self.api.set_speed(0)

        # Let the robot settle at rest before control begins.
        for _ in range(self.settle_steps):
            time.sleep(self.dt)
            location = self.api.get_location()
            heading = self._heading_deg_to_rad(self.api.get_heading())
            speed = (self.api.get_speed() / 255.0) * self.vel_limit

            if isinstance(location, dict):
                x0 = float(location.get("x", 0.0)) / 100.0
                y0 = float(location.get("y", 0.0)) / 100.0
            else:
                x0, y0 = 0.0, 0.0

            self.state_odom = np.array([x0, y0, heading, speed], dtype=np.float32)
            self.state_true = self.state_odom.copy()

        obs = self._get_observation()
        self.last_obs = obs.copy()

        info = {
            "state_true": self.state_true.copy(),
            "state_odom": self.state_odom.copy(),
        }

        return obs, info

    def _sense_collision(self) -> bool:
        """
        Detect collisions from live Sphero sensor readings.

        Uses acceleration jerk, velocity drop, and gyroscope spikes. Updates
        self.last_collision_flag before _get_observation() is called so obs[4]
        reflects the current step's collision state.
        """
        acceleration = self.api.get_acceleration()
        velocity = self.api.get_velocity()
        gyroscope = self.api.get_gyroscope()

        if acceleration is None or velocity is None or gyroscope is None:
            self.last_collision_flag = 0.0
            self._last_collision_debug = {}
            return False

        ax = float(acceleration.get("x", 0.0))
        ay = float(acceleration.get("y", 0.0))
        az = float(acceleration.get("z", 0.0))
        accel_mag = float(np.sqrt(ax * ax + ay * ay + az * az))

        vx = float(velocity.get("x", 0.0))
        vy = float(velocity.get("y", 0.0))
        speed_cm_s = float(np.sqrt(vx * vx + vy * vy))

        gx = float(gyroscope.get("x", gyroscope.get("pitch", 0.0)))
        gy = float(gyroscope.get("y", gyroscope.get("roll", 0.0)))
        gz = float(gyroscope.get("z", gyroscope.get("yaw", 0.0)))
        gyro_mag = float(np.sqrt(gx * gx + gy * gy + gz * gz))

        if self._last_accel_mag is None:
            self._last_accel_mag = accel_mag
            self._last_speed_cm_s = speed_cm_s
            self._last_gyro_mag = gyro_mag
            self.last_collision_flag = 0.0
            self._last_collision_debug = {
                "accel_mag": accel_mag,
                "accel_jerk": 0.0,
                "speed_cm_s": speed_cm_s,
                "speed_drop_cm_s": 0.0,
                "gyro_mag": gyro_mag,
                "gyro_spike": 0.0,
            }
            return False

        accel_jerk = abs(accel_mag - float(self._last_accel_mag))
        speed_drop = float(self._last_speed_cm_s) - speed_cm_s
        gyro_spike = abs(gyro_mag - float(self._last_gyro_mag))
        now = time.time()

        moving_collision = (
            accel_jerk > self.collision_jerk_threshold
            and accel_mag > self.collision_accel_threshold
            and speed_drop > self.collision_speed_drop_threshold
        )
        stationary_impact = (
            speed_cm_s < 5.0
            and accel_jerk > self.stationary_collision_jerk_threshold
        )
        rotation_collision = (
            gyro_spike > self.collision_gyro_spike_threshold
            and gyro_mag > self.collision_gyro_mag_threshold
            and accel_mag > 1.2
        )

        collision = bool(
            (moving_collision or stationary_impact or rotation_collision)
            and (now - self._last_collision_time > self.collision_dead_time)
        )

        self._last_accel_mag = accel_mag
        self._last_speed_cm_s = speed_cm_s
        self._last_gyro_mag = gyro_mag

        self._last_collision_debug = {
            "accel_mag": accel_mag,
            "accel_jerk": accel_jerk,
            "speed_cm_s": speed_cm_s,
            "speed_drop_cm_s": speed_drop,
            "gyro_mag": gyro_mag,
            "gyro_spike": gyro_spike,
        }

        if collision:
            self._last_collision_time = now
            self.last_collision_flag = 1.0
            return True

        self.last_collision_flag = 0.0
        return False

    def step(self, action: np.ndarray):
        """Execute one step with the real robot.

        Args:
            action: Action array [speed, heading].

        Returns:
            Observation, reward, terminated, truncated, info.
        """
        assert self.state_odom is not None, "Call reset() before step()."

        # By default, logged setpoint tracks goal_pos. Wrappers/controllers can
        # override this per-step via set_current_setpoint().
        if not self._use_custom_setpoint:
            self.current_setpoint = np.asarray(self.goal_pos, dtype=np.float32).copy()

        # Clip action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        speed_cmd, heading_cmd = float(action[0]), float(action[1])
        heading_cmd = self._wrap_angle(heading_cmd)

        # Convert to robot command
        if speed_cmd < 0:
            heading_deg = int(np.degrees(self._wrap_angle(heading_cmd + np.pi)) % 360)
            speed_raw = int(np.clip(-speed_cmd/self.vel_limit * 255, 0, 255))
        else:
            heading_deg = int(np.degrees(heading_cmd) % 360)
            speed_raw = int(np.clip(speed_cmd / self.vel_limit * 255, 0, 255))

        # Send command
        self.set_heading_and_speed(heading_deg, speed_raw)
        self.last_command = (heading_cmd, speed_cmd, time.time())

        location = self.api.get_location()
        heading = self._heading_deg_to_rad(self.api.get_heading())
        speed = (self.api.get_speed() / 255.0) * self.vel_limit

        if isinstance(location, dict):
            x_new = float(location.get("x", 0.0))/100.0  # Convert cm to m
            y_new = float(location.get("y", 0.0))/100.0
        else:
            x_new, y_new = 0.0, 0.0

        self.state_odom = np.array([x_new, y_new, heading, speed], dtype=np.float32)
        self.state_true = self.state_odom.copy()

        collision_detected = self._sense_collision()

        self.step_count += 1

        terminated = False
        truncated = self.step_count >= self.max_steps

        obs = self._get_observation()
        self.last_obs = obs.copy()

        info: Dict[str, Any] = {
            "state_true": self.state_true.copy(),
            "state_odom": self.state_odom.copy(),
            "collision": collision_detected,
            "collision_flag": self.last_collision_flag,
            "collision_debug": self._last_collision_debug.copy(),
            "acceleration": self.api.get_acceleration(),
            "gyroscope": self.api.get_gyroscope(),
            "orientation": self.api.get_orientation(),
            "velocity": self.api.get_velocity(),
            "speed_cmd": speed_cmd,
            "heading_cmd": heading_cmd,
            "setpoint_xy": self.current_setpoint.copy(),
        }

        # Log each control step once; estimator updates are handled separately.
        self.vis.record(
            gt_state=self.state_true,
            odom_state=self.state_odom,
            action=np.array([speed_cmd, heading_cmd], dtype=np.float32),
            est_state=self.latest_est_mean,
            est_cov=self.latest_est_cov,
            setpoint=self.current_setpoint,
        )

        return obs, None, terminated, truncated, info
    
    # ---- Interface compatibility methods ---- #

    def _get_observation(self) -> np.ndarray:
        """
        Construct observation from odometry state plus measurement noise.
        """
        assert self.state_odom is not None
        rng = self.np_random

        x_o, y_o, heading_o, speed_o = self.state_odom

        x_meas = x_o + rng.normal(0.0, self.obs_noise_std_pos)
        y_meas = y_o + rng.normal(0.0, self.obs_noise_std_pos)
        heading_meas = self._wrap_angle(heading_o + rng.normal(0.0, self.obs_noise_std_vel))
        speed_meas = speed_o + rng.normal(0.0, self.obs_noise_std_vel)

        obs = np.array(
            [x_meas, y_meas, heading_meas, speed_meas, self.last_collision_flag],
            dtype=np.float32,
        )
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def get_true_state(self) -> np.ndarray:
        """
        Return a copy of the state [x, y, heading, speed].
        For real robot, this is the odometry estimate.
        """
        assert self.state_true is not None
        return self.state_true.copy()

    def get_odom_state(self) -> np.ndarray:
        """
        Return a copy of the odometry state [x, y, heading, speed].
        """
        assert self.state_odom is not None
        return self.state_odom.copy()

    def render(self):
        """
        Delegate rendering to the shared visualiser.
        """
        self.vis.render(self.state_true, self.state_odom)

    def close(self):
        """
        Clean up: stop the robot and close connection if needed.
        """
        self.api.set_speed(0)
        self.vis.close()

    # ---- Robot-specific control methods ---- #

    def set_heading_and_speed(self, heading_deg: float, speed: int):
        """
        Send heading and speed command to real robot.
        
        Args:
            heading_deg: Heading in degrees (0=forward, 90=right, 180=backward, 270=left)
            speed: Speed command in [0, 255]
        """
        with self._lock:
            self.api.set_heading(int(heading_deg))
            self.api.set_speed(int(np.clip(speed, 0, 255)))

    def emergency_stop(self):
        """
        Emergency stop: set speed to 0.
        """
        with self._lock:
            self.api.set_speed(0)

    def set_speed(self, speed: int):
        """
        Set speed directly.
        
        Args:
            speed: Speed command in [0, 255]
        """
        with self._lock:
            self.api.set_speed(int(np.clip(speed, 0, 255)))

    def set_heading(self, heading_deg: float):
        """
        Set heading directly.
        
        Args:
            heading_deg: Heading in degrees
        """
        with self._lock:
            self.api.set_heading(int(heading_deg))
