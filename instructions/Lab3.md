# Lab 3: Path planning

In Lab 3, you need to implement an A* path planner (or something else, eg. RRT, RL) to escape a known maze.

## Key learning outcomes

- Understand and implement a global planner into a robot navigation stack.
- Interface this with low level controllers and estimators from your previous labs to escape a maze.

## What you need to do

- **Implement a global waypoint planner** to escape a grid like mazeworld.
- **Add a lower level controller** to move between these waypoints.
- **Add replanning as required** to ensure the robot escapes a known maze.

The lab3.py skeleton is pre-configured with code to generate a simulated maze matching our labs, with the end goal as a target.

## Submission requirements

- Submit a CSV result output for assessment. It should comprise all steps from start to end of a run, or to within a specified goal tolerance.
- During assessment, you will be evaluated on how well your robot navigates in a maze.

## Automark assessment

The automarker reads a CSV log from `studentid_lab3.csv`.

### Required CSV columns

| Column | Description |
| --- | --- |
| `sim_x` | Simulator x position |
| `sim_y` | Simulator y position |
| `real_x` | Estimated x position |
| `real_y` | Estimated y position |

### Evaluated metrics and pass thresholds

| Metric | Default threshold |
| --- | --- |
| Final distance to target (real) | ≤ 0.10 m | 
| Distance to optimal path (real) | ≤ 0.20 m | 
| Final distance to target (sim) | ≤ 0.10 m |
| Distance to optimal path (sim) | ≤ 0.20 m | 

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

## Setting yourself up for Lab 4

- Lab 4 requires you to reproduce this lab using a learning based approach
- You could use Lab 3 to generate training data, or go back to lab 2 and use teleoperation.

---
