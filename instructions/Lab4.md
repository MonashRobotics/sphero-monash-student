# Lab 4: Learning-based navigation in a known environment

In Lab 4, you need to implement a learning based approach (RL, Behaviour cloning, ilqr with learned dynamics) to escape a known maze.

## Key learning outcomes

- Understand and implement learning in a robot navigation stack.
- Familiarise yourself with the process to collect data to train a model, and common success of failure modes associated with learning.

## What you need to do

- **Collect data** to train a suitable model.
- **Deploy the model** to move the robot to the goal as fast as possible.
- **Add fail safes and safety layers** to ensure the learned policy behaves.

The lab4.py skeleton is pre-configured with code to generate a simulated maze matching our labs, with the end goal as a target.

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

## Setting yourself up for the Project

- You now have the basics of a functional robot navigation stack.
- For your final project, you will be required to build on this in a competition to race other robots to escape an unknown maze. 

---
