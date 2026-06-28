# sphero-monash

This repository contains the labs and codebase for working with Sphero robots at Monash. The code is built on a sphero-unsw interface.

Throughout this unit, you will modify and extend this codebase by adding functionality across a sequence of labs. By the end of the unit, you will have built enough capability to complete the final project.

Each lab has two components:

1. **Automarked individual assessment**
2. **Team demo on a real robot**

Your mark is a combination of both components.

The final project is submitted in teams and also consists of both a simulated demo and a real robot demo. Time on the real robots is intentionally limited — the goal is for you to conquer the **sim-to-real gap**.

## Robot Interface

Both the real robot and the provided simulator implement a [`gymnasium`](https://gymnasium.farama.org/index.html)-like interface.

### Action Space

The robot action is defined as:

```python
action = [speed, heading_angle]
```

| Index | Variable        | Description           |
| ----: | --------------- | --------------------- |
|   `0` | `speed`         | Desired robot speed   |
|   `1` | `heading_angle` | Desired heading angle |

### Observation Space

The robot observation is defined as:

```python
observation = [
    x_meas,
    y_meas,
    heading_meas,
    speed_meas,
    collision_flag,
]
```

| Index | Variable         | Description            |
| ----: | ---------------- | ---------------------- |
|   `0` | `x_meas`         | Measured x-position    |
|   `1` | `y_meas`         | Measured y-position    |
|   `2` | `heading_meas`   | Measured heading angle |
|   `3` | `speed_meas`     | Measured speed         |
|   `4` | `collision_flag` | Collision indicator    |

### Robot `info` dictionary

Each call to `robot.step(action)` returns an `info` dictionary with extra diagnostic details from the real Sphero robot.

```python
obs, reward, terminated, truncated, info = robot.step(action)
```

The main fields are:

| Field             |              Units | Description                                                                                                                            |
| ----------------- | -----------------: | -------------------------------------------------------------------------------------------------------------------------------------- |
| `state_true`      | `[m, m, rad, -]` | Current robot state `[x, y, heading, speed]`. For the real robot this is the same as odometry.                                         |
| `state_odom`      | `[m, m, rad, -]` | Odometry-based robot state `[x, y, heading, speed]`. Location from the Sphero API is converted from centimetres to metres.             |
| `collision`       |            boolean | Whether a collision was detected on this step.                                                                                         |
| `collision_flag`  |     `0.0` or `1.0` | Numeric collision flag. This is also included as the final observation value, `obs[4]`.                                                |
| `collision_debug` |              mixed | Extra values used by the software collision detector, useful for threshold tuning. See below for units.                                |
| `acceleration`    |                `g` | Latest acceleration vector from the Sphero accelerometer. `1 g = 9.80665 m/s²`; each axis is approximately in the range `-8` to `8 g`. |
| `orientation`     |            degrees | Latest robot orientation/attitude. `pitch` and `yaw` are approximately `-180°` to `180°`; `roll` is approximately `-90°` to `90°`.     |
| `gyro`            |          degrees/s | Latest gyroscope rate reading. Axes are approximately in the range `-2000°/s` to `2000°/s`.                                            |
| `velocity`        |               cm/s | Latest velocity vector from the motor encoders. `x` is right/left velocity and `y` is forward/back velocity.                           |
| `speed_cmd`       |                m/s | Speed command sent through the environment action interface. This is converted internally to the Sphero speed scale.                   |
| `heading_cmd`     |            radians | Heading command sent through the environment action interface. This is converted internally to degrees for the Sphero API.             |
| `setpoint_xy`     |           `[m, m]` | Current target/setpoint position used for logging and visualisation.                                 

Example:

```python
obs, reward, terminated, truncated, info = robot.step(action)

if info["collision"]:
    print("Collision detected")
    print(info["collision_debug"])
```

The collision detector is software-based. It uses changes in acceleration, drops in velocity, and gyroscope spikes to estimate when the Sphero has hit an obstacle.

`collision_debug` contains values such as:

| Debug value          |                Units | Description                                                           |
| -------------------- | -------------------: | --------------------------------------------------------------------- |
| `accel_mag`          |                  `g` | Magnitude of the acceleration vector.                                 |
| `accel_jerk`         |           `g / step` | Change in acceleration magnitude since the previous control step.     |
| `speed_cm_s`         |               `cm/s` | Magnitude of the Sphero velocity vector.                              |
| `speed_drop`         |      `cm/s per step` | Drop in speed since the previous control step.                        |
| `gyro_mag`           |          `degrees/s` | Magnitude of the gyroscope rate vector.                               |
| `gyro_spike`         | `degrees/s per step` | Change in gyroscope magnitude since the previous control step.        |
| `moving_collision`   |              boolean | Collision condition based on acceleration jerk and speed drop.        |
| `stationary_impact`  |              boolean | Collision condition for impacts while the robot is nearly stationary. |
| `rotation_collision` |              boolean | Collision condition based on a sudden gyroscope spike.                |


### Goals

The environment can be configured with a 2d positional goal and goal tolerance:
* `env.goal_pos` 
* `env.goal_tolerance`

### Teleoperation example

An example using the simulator and robot is available in:
`examples/teleop.py`

## Lab 1 - PID Control, Calibration, and Simulation

In Lab 1, you will design PID controllers to drive your robot to a given pose set-point.

A simulator environment is provided so that you can develop and test your controller in simulation first. You will also need to override the simulator dynamics model so that it matches your real robot’s behaviour as closely as possible.

This calibration is **very important** — it will make or break the performance of your robot in all remaining labs.

[Click here for Lab 1 instructions.](./instructions/Lab1.md)

## Lab 2 - EKF Odometry

In Lab 2, you will use a motion model of the robot together with measurements to improve the robot’s odometry estimate.

For the lab demo, you will move through a sequence of points and report on your estimation accuracy.

Again, calibration between the simulator, model, and real robot is key to performance. The more closely the real robot behaves like the simulator, the more likely your solution is to work in the real demo.

[Click here for Lab 2 instructions.](./instructions/Lab2.md)

## Lab 3 - Path Planning

In Lab 3, you will discretise the robot’s operating space and investigate motion planning in a known environment.

Performance in this task will depend heavily on how well your Lab 2 localisation system worked.

[Click here for Lab 3 instructions.](./instructions/Lab3.md)

## Lab 4 - Navigation in a Known Environment

In Lab 4, you will implement a learning-based method to find a sequence of moves that helps your robot navigate around a race track.

[Click here for Lab 4 instructions.](./instructions/Lab4.md)

## Final Demo

For the final demo, you will program your robot to race another robot in an unknown environment, without knowing the environment layout beforehand.

You may consider training a reinforcement learning approach, or chaining together planning, mapping, and control modules.

[Click here for final demo instructions.](./instructions/Project.md)

---
# Installation

This project requires Python 3.9 or newer.

We recommend creating a virtual environment before installing the package and its dependencies.

## Option 1: Create an environment with `venv`

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade `pip` and install the project:

```bash
python -m pip install --upgrade pip
pip install -e .
```

If you want the extra teaching and notebook dependencies from `requirements.txt`, install them as well:

```bash
pip install -r requirements.txt
```

## Option 2: Create an environment with `uv`

Create a virtual environment and activate it:

```bash
uv venv
source .venv/bin/activate
```

Install the project into that environment:

```bash
uv pip install -e .
```

To install the extra dependencies listed in `requirements.txt`:

```bash
uv pip install -r requirements.txt
```

## Verify the installation

You can check that the package imports correctly with:

```bash
python -c "import sphero_env; print('sphero_env import OK')"
```

You can then run the teleoperation example with:

```bash
python examples/teleop.py
```

## Connecting to a real sphero

1. Gently shake the sphero robot - it should light up and show a unique bluetooth address on it.
2. Place it on the ground (it must not be moving when you connect)
3. When you run a robot connection (eg. using the teleop code), you will be asked to confirm the sphero bluetooth address to connect.

