# Lab 3: Occupancy Grid Mapping

Lab 3 focuses on discretising the robot workspace and building a contact-based simultaneous localisation and occupancy grid mapping solution.

## Key learning outcomes:
- Represent a continuous robot workspace as a discrete occupancy grid
- Implement contact-based simultaneous localisation and mapping (SLAM) concepts
- Analyse how localisation quality influences mapping performance

You need to do the following things:
-
- **Discretise the operating environment** into an occupancy grid representation.
- **Implement contact-based mapping updates** to infer occupied and free space from robot interactions.
- **Integrate localisation and mapping** so the robot can maintain an accurate map while estimating pose.
- **Evaluate mapping quality** with respect to the performance of your Lab 2 odometry/localisation pipeline.

Submission requirements:
-
- Submit your CSV result output for assessment.
- You will be evaluated on mapping and localisation performance in the discretised environment.

## Automark assessment

The automarker reads a CSV log from `logs/lab3.csv` (override with the `LAB3_LOG` environment variable).  
See `tests/test_automark_metrics.py` (`test_lab3_automark_metrics_from_log`) for the full evaluation logic.

### Required CSV columns

| Column | Description |
| --- | --- |
| `gt_x` | Ground-truth x position (m) |
| `gt_y` | Ground-truth y position (m) |
| `est_x` | Estimated x position (m) |
| `est_y` | Estimated y position (m) |

### Evaluated metrics and pass thresholds

| Metric | Default threshold | Override env variable |
| --- | --- | --- |
| Localisation RMSE | ≤ 0.25 m | `LAB3_MAX_LOCALISATION_RMSE_M` |

Setting yourself up for Lab 4:
-
* Lab 4 will use the discrete map and localisation outputs from this lab.
* Better map quality will directly improve navigation and planning performance.
* Ensure your representation supports path planning across the race-track environment.
