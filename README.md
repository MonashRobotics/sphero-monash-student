# sphero-monash

This repository holds the labs and codebase for sphero robots at monash. You will be modifying and editing this code-base throughout this unit, adding functionality until you have built up enough code to complete the final project for the unit.

Each of the labs is divided into two components - an automarked individual assessment, and a team demo on a real robot. Your mark is a combination of both. The final project is submitted in teams, and also consists of a simulated and real robot demo. Time on the real robots is intentionally limited - I want you to conquer the *sim-to-real* gap.

### Lab 1 - PID Control, calibrating your robot and the simulator

As a first task, you will be tasked with designing PID controllers to drive your robot to a given pose set-point. We have provided a simulator "environment" to do this in simulation first, and you will need to override the simulator dynamics model to match your robot behaviour as closely as possible. This calibration is **very important** - it will make or break the performance of your robot for all remaining labs.

[Click here for Lab 1 instructions.](./instructions/Lab1.md)

### Lab 2 - EKF Localisation

In your second lab, you will be required to use a motion model of the robot together with contact measurements to improve the odometry estimate of the robot. In the lab demo, you will need to move through a sequence of points, and report on your estimation accuracy. Again, calibrating the simulator and model with the real robot is key to performance. The closer the real robots behaviour resembles the simulator, the more likely your work is to work on a real demo.

[Click here for Lab 2 instructions.](./instructions/Lab2.md)

### Lab 3 - A* Path planning

In your third lab, we will discretise the space the robot operates in, and investigate A* motion planning in a known environment. Performance in this task will be very much dependent on the degree to which your lab 2 worked.

[Click here for Lab 3 instructions.](./instructions/Lab3.md)

### Lab 4 - Navigation in a known environment

In lab 4, you will be required to implement a learning-based method to find a sequence of moves to help your robot move around a race track.

[Click here for Lab 4 instructions.](./instructions/Lab4.md)

### Final demo

As part of your final demo, you will need to program your robot to race another robot in an unknown environment, without knowing what this looks like before hand. You could consider training an RL approach to do this, or chaining together a set of planning, mapping and control modules.

[Click here for final demo instructions.](./instructions/Project.md)
