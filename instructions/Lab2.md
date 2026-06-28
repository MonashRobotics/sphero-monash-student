# Lab 2: EKF Odometry

In Lab 2, you should use an EKF to improve the vehicle odometry by incorporating sensor measurements. You goal should also be to produce a well calibrated uncertainty model that ensures that the true robot position is always within your prediction.

## Key learning outcomes

- Understand how to implement and apply an EKF that fuses robot sensor measurements with a suitable dynamics model.
- Evaluate how measurement quality and process noise tuning affect localisation accuracy.

## What you need to do

- **Use your robot motion model** from Lab 1 as the process model for odometry prediction.
- **Implement a suitable measurement model** `h(x)`: using one or more of the following measurements (robot speed, robot heading, robot collision) in simulation. On the real robot, you also have access to a gyroscope, accelerometer and individual wheel velocities of the differential drive robot inside the Sphero Bolt ball. 
- **Implement an EKF** using the two models above.
- **Tune `Q` (process noise) and `R` (measurement noise)** so that the EKF estimate tracks ground truth well.
- **Visualise** the uncertainty using the provided code.
- **Calibrate** the simulator dynamics and noise parameters against the real robot.

The base lab2 file is configured for robot teleoperation to allow you to test the filter.

## Submission requirements

- Submit a CSV result output for assessment. It should comprise 200 steps of robot motion.
- During assessment, you will be evaluated on how well the EKF estimate tracks
  the ground-truth position as the robot navigates in the arena.

## Automark assessment

The automarker reads a CSV log from `studentid_lab2.csv`.

### Required CSV columns

| Column | Description |
| --- | --- |
| `sim_x` | Simulator (ground-truth) x position |
| `sim_y` | Simulator (ground-truth) y position |
| `real_x` | EKF mean estimated x position |
| `real_y` | EKF mean estimated y position |
| `P_xx` | Posterior covariance P (x,x) |
| `P_xy` | Posterior covariance P (x,y) / (y,x) |
| `P_yy` | Posterior covariance P (y,y) |

### Evaluated metrics and pass thresholds

| Metric | Default threshold |
| --- | --- | --- |
| Mean Mahalanobis distance of position error (`sim - real`) | ≤ 4.0 | 
| Chi-square pass rate (2 DoF, 95 % gate) | ≥ 0.90 | 

### Guide to metrics and tuning

In an EKF, Mahalanobis distance measures how far a sensor measurement is from the filter's prediction, scaled by the filter's expected uncertainty. This squared distance is called the Normalized Innovation Squared (NIS):

$$
  \text{NIS}=\nu _{k}^{T}S_{k}^{-1}\nu _{k}
$$

If your noise matrices `Q` and `R` are perfectly calibrated, errors behave like white noise. Mathematically, the sum of squared white noise variables follows a Chi-squared distribution:

$$\text{NIS}\sim \chi _{p}^{2}$$
where \(p\) = degrees of freedom = number of variables in your measurement vector-.

#### How to Tune Your EKF

Plot your NIS values against the theoretical $\chi ^{2}$ upper and lower bounds (e.g., a 95% confidence interval):

* NIS is ABOVE the bounds (Fail High): The filter is Optimistic (overconfident). Real errors are larger than expected. You should probably increase Process Noise `Q` or Measurement Noise `R`.
* NIS is BELOW the bounds (Fail Low): The filter is Pessimistic (sluggish). Real errors are smaller than expected. You should probably decrease Process Noise `Q` or Measurement Noise `R`.
* NIS is INSIDE the bounds: The filter is well calibrated.

## Setting yourself up for Lab 3

- Lab 3 depends on reliable localisation from this lab.
- Good EKF tuning here directly improves Lab 3 navigation.

---
