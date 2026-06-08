# Lab 2: EKF Localisation in a Known Map (Contact-Based Pseudo-Lidar)

Lab 2 extends robot odometry with contact-based range measurements to achieve
accurate localisation inside a known maze.  The robot is teleoperated and
constrained to move in the four cardinal directions (North, East, South, West)
so that every collision with a wall provides an unambiguous 1-D range
measurement — a pseudo-lidar reading without any extra sensor hardware.

## Key learning outcomes

- Implement an EKF that fuses **odometry** (process model) with **range-to-wall** measurements (measurement model) in a known occupancy map.
- Understand how restricting motion to cardinal directions turns collision events into informative pseudo-lidar observations.
- Evaluate how measurement quality and process noise tuning affect localisation accuracy.
- Experience simulator-to-real transfer for a contact-based localisation system.

## What you need to do

- **Teleop the robot** using arrow keys or WASD — the robot is restricted to N/E/S/W headings only.
- **Use your robot motion model** from Lab 1 as the process model for odometry prediction.
- **Implement the range measurement model** `h(x)`: given the EKF state `[x, y, heading, speed]` and the known occupancy map, ray-cast to find the expected distance to the nearest wall in the current heading direction.
- **Fuse range measurements** into the EKF whenever the robot collides with a wall: the measured range is the distance traveled since the last heading change.
- **Tune `Q` (process noise) and `R_range` (range noise variance)** so that the EKF estimate converges and tracks ground truth well.
- **Calibrate** the simulator dynamics and noise parameters against the real robot.

## How a range measurement is generated

```
                 EKF state x = [x, y, heading, speed]
                        │
                 ray_cast(known_map, x, y, heading)
                        │
               h(x) = expected distance to wall
```

When the robot hits a wall:

```
z_measured = ‖odom_position_now − odom_position_at_last_heading_change‖
```

The EKF update compares `z_measured` with `h(x)` (scalar innovation) to correct
the position estimate.

## Controls

| Key | Action |
| --- | --- |
| `↑` / `W` | Move North |
| `→` / `D` | Move East |
| `↓` / `S` | Move South |
| `←` / `A` | Move West |
| `SPACE` | Emergency stop |
| `Q` / `ESC` | Quit |

## Running

```bash
# Simulation
python labs/Lab2.py --sim

# Real robot
python labs/Lab2.py
```

## Submission requirements

- Submit your CSV result output for assessment.
- During assessment, you will be evaluated on how well the EKF estimate tracks
  the ground-truth position as the robot navigates the known maze.

## Automark assessment

The automarker reads a CSV log from `logs/lab2.csv` (override with the `LAB2_LOG` environment variable).  
See `tests/test_automark_metrics.py` (`test_lab2_automark_metrics_from_log`) for the full evaluation logic.

### Required CSV columns

| Column | Description |
| --- | --- |
| `sim_x` | Simulator (ground-truth) x position |
| `sim_y` | Simulator (ground-truth) y position |
| `real_x` | EKF mean estimated x position |
| `real_y` | EKF mean estimated y position |
| `target_x` | Current target x position |
| `target_y` | Current target y position |
| `P_xx` | Posterior covariance P (x,x) |
| `P_xy` | Posterior covariance P (x,y) / (y,x) |
| `P_yy` | Posterior covariance P (y,y) |

### Evaluated metrics and pass thresholds

| Metric | Default threshold | Override env variable |
| --- | --- | --- |
| Mean Mahalanobis distance of position error (`sim - real`) | ≤ 4.0 | `LAB2_MAX_MEAN_MAHALANOBIS` |
| Chi-square pass rate (2 DoF, 95 % gate) | ≥ 0.90 | `LAB2_MIN_CHI_SQUARE_PASS_RATE` |
| Posterior covariance validity (symmetric positive semidefinite) | all eigenvalues ≥ −1 × 10⁻¹⁰ | — |

## Setting yourself up for Lab 3

- Lab 3 depends on reliable localisation from this lab.
- Good EKF tuning here directly improves Lab 3 occupancy mapping and DBF localisation.
- Keep your simulator and real robot behaviour aligned to support discrete-space localisation in Lab 3.

---

## 1. State Representation

The state vector $\mathbf{x}_k$ at time step $k$ tracks the robot's 2D position and its continuous heading angle $\theta$. Even though the robot snaps to cardinal directions, tracking $\theta$ as a continuous variable allows the EKF to handle uncertainty during transitions.

$$\mathbf{x}_k = \begin{bmatrix} x_k \\ y_k \\ \theta_k \end{bmatrix}$$

The covariance matrix $\mathbf{P}_k$ represents the uncertainty of this state estimate.

---

## 2. Motion Model (Prediction Step)

The robot executes an action by turning on the spot by $\Delta \theta_k$ to align with a cardinal direction, and then moving forward a distance $d_k$. 

### Process Equations
The non-linear state transition function $\mathbf{f}(\mathbf{x}_{k-1}, \mathbf{u}_k)$ given the control input $\mathbf{u}_k = [d_k, \Delta \theta_k]^T$ is modeled as:

$$\mathbf{x}_k = \mathbf{f}(\mathbf{x}_{k-1}, \mathbf{u}_k) + \mathbf{w}_k = \begin{bmatrix} x_{k-1} + d_k \cos(\theta_{k-1} + \Delta \theta_k) \\ y_{k-1} + d_k \sin(\theta_{k-1} + \Delta \theta_k) \\ \theta_{k-1} + \Delta \theta_k \end{bmatrix} + \mathbf{w}_k$$

Where $\mathbf{w}_k \sim \mathcal{N}(0, \mathbf{Q}_k)$ represents the Gaussian process noise.

### Linearization (Jacobian)
To propagate state uncertainty, we calculate the Jacobian matrix $\mathbf{F}_x$ with respect to the state $\mathbf{x}$:

$$\mathbf{F}_x = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} 1 & 0 & -d_k \sin(\theta_{k-1} + \Delta \theta_k) \\ 0 & 1 & d_k \cos(\theta_{k-1} + \Delta \theta_k) \\ 0 & 0 & 1 \end{bmatrix}$$

### EKF Predict Equations
Using these equations, the state estimate and its covariance are projected forward in time:

$$\hat{\mathbf{x}}_k^- = \mathbf{f}(\hat{\mathbf{x}}_{k-1}, \mathbf{u}_k)$$

$$\mathbf{P}_k^- = \mathbf{F}_x \mathbf{P}_{k-1} \mathbf{F}_x^T + \mathbf{Q}_k$$

---

## 3. Measurement Model (Update Step)

The update step is purely event-driven and triggers **only** when a collision is detected. At the exact moment of collision, the actual distance to the obstacle is known to be zero ($z_k = 0$).

### Measurement Equation
Let $O(\mathbf{x})$ be a lookup function that queries the grid map and returns the global 2D coordinates of the obstacle $[x_{obs}, y_{obs}]^T$ directly in front of the robot based on its current position and heading.

The measurement function $h(\mathbf{x}_k)$ calculates the expected distance to that obstacle:

$$h(\mathbf{x}_k) = \sqrt{(x_{obs} - x_k)^2 + (y_{obs} - y_k)^2}$$

Because this update only runs when the robot physically hits a wall, the actual observation is a constant:
$$z_k = 0$$

### Linearization (Jacobian)
We compute the Jacobian matrix $\mathbf{H}$ with respect to the state $\mathbf{x}$, evaluated at the predicted state $\hat{\mathbf{x}}_k^-$:

$$\mathbf{H} = \frac{\partial h}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial h}{\partial x} & \frac{\partial h}{\partial y} & \frac{\partial h}{\partial \theta} \end{bmatrix}$$

$$\frac{\partial h}{\partial x} = \frac{-(x_{obs} - \hat{x}_k^-)}{h(\hat{\mathbf{x}}_k^-)}, \quad \frac{\partial h}{\partial y} = \frac{-(y_{obs} - \hat{y}_k^-)}{h(\hat{\mathbf{x}}_k^-)}, \quad \frac{\partial h}{\partial \theta} = 0$$

> **Note on Heading Derivative:** $\frac{\partial h}{\partial \theta} = 0$ because the lookup function treats the discrete cardinal direction as a fixed line of sight. Small continuous variations in $\theta$ do not change the distance to a grid wall that is perpendicular to the travel axis.

---

## 4. EKF Update Execution

When the collision sensor triggers, execute the following correction steps:

1. **Compute Measurement Innovation ($\mathbf{y}_k$):**
   $$\mathbf{y}_k = z_k - h(\hat{\mathbf{x}}_k^-) = 0 - h(\hat{\mathbf{x}}_k^-)$$
   *(This heavily penalizes the state estimate if the map lookup expected the wall to be further away).*

2. **Compute Innovation Covariance ($\mathbf{S}_k$):**
   $$\mathbf{S}_k = \mathbf{H} \mathbf{P}_k^- \mathbf{H}^T + \mathbf{R}_k$$
   *(Where $\mathbf{R}_k$ is the measurement noise variance, reflecting the structural "softness" or sensor uncertainty of the collision event).*

3. **Compute Kalman Gain ($\mathbf{K}_k$):**
   $$\mathbf{K}_k = \mathbf{P}_k^- \mathbf{H}^T \mathbf{S}_k^{-1}$$

4. **Update State Estimate ($\hat{\mathbf{x}}_k$):**
   $$\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k^- + \mathbf{K}_k \mathbf{y}_k$$

5. **Update Estimate Covariance ($\mathbf{P}_k$):**
   $$\mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_k^-$$