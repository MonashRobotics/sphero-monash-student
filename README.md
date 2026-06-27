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

## Lab 3 - A* Path Planning

In Lab 3, you will discretise the robot’s operating space and investigate A* motion planning in a known environment.

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

