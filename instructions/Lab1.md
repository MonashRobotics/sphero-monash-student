# Lab 1: PID Control, calibrating your robot and the simulator

Lab 1 will focus on familiarising yourself with the robot and simulator, teleoperating it and understanding the action and state space of the robot.

## Key learning outcomes:
- Implement and tune a PD controller driving a robot to a fixed robot pose
- Understand robot state representations and the value of good modelling
- Appreciate the sim2real gap - in practice, we never have real world time (robot batteries only last so long, real world experiments are complex and time consuming to set up), and need to develop in simulation. This requires good simulators and the quality of your simulator and dynamics models will affect future robot performance.

You need to do the following things:
-
- **Understand the codebase**: Explore and understand the codebase provided, in particular using the simulator, teleoperation scripts and data logging capabilities provided.
- **Calibrate your simulator** so that it matches your robot. We provide a crude robot dynamics model, but you need to tune the parameters to make it closer to the real robot. If you like, you can also replace the dynamics model with something more sophisticated, by drawing on the workshop theory.
  - You should use the logging capabilities in the repository to log both the robot and simulator state data. Plotting these on the same chart will assist with calibration.
- **Implement a PID controller** to move the robot to a specific target location. Tune the controller to get a good response.
  
Submission requirements:
-
- Use the provided logging functions to save data. Combine both your simulated and real robot data into a csv file `Lab1.csv` containing robot/sim state information over 100 time steps as you command the robot to move to the point (0.5m,0.5m) relative to the starting location of the robot.
- For this lab, you will be evaluated on the final positioning accuracy of the robot (important for the real world demo), and how closely the simulator states match the real robot recordings.


The automarker reads a CSV log from `studentid_lab1.csv`.

### Required CSV columns

| Column | Description |
| --- | --- |
| `sim_x` | Simulator x position (m) |
| `sim_y` | Simulator y position (m) |
| `real_x` | Real robot x position (m) |
| `real_y` | Real robot y position (m) |

### Evaluated metrics and pass thresholds

| Metric | Default threshold |
| --- | --- |
| Final distance to target (real) | ≤ 0.10 m | 
| Sim-vs-real trajectory RMSE | ≤ 0.20 m | 
| Final distance to target (sim) | ≤ 0.10 m |

Setting yourself up for Lab 2:
-
* Lab 2 will assume that your robot can move to target poses. 
* It will also assume that your simulator closely matches the real robot behaviour. Aim to do much better than the targets above.
* You will need to understand the robot dynamics model, as you will be using it to improve the robot odometry.
