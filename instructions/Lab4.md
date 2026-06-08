# Lab 4: Navigation in a known environment

Lab 4 focuses on planning and control to navigate a known race-track environment using your discrete-space localisation and mapping pipeline.

## Key learning outcomes:
- Implement a navigation pipeline that plans and executes paths in known environments
- Compare planning approaches such as classical search, behaviour cloning, and reinforcement learning
- Integrate localisation, mapping, and control into end-to-end autonomous navigation

You need to do the following things:
-
- **Use your Lab 3 discrete-space localisation** to determine robot position in the race track.
- **Implement a planner** to generate a sequence of target poses for navigating through the race track.
- **Execute planned trajectories** using your existing control stack.
- **Evaluate planning strategy performance** using one of the recommended approaches (for example A*, bug algorithms, behaviour cloning, or reinforcement learning).

Submission requirements:
-
- Submit your CSV result output for assessment.
- You will be evaluated on successful navigation performance in the known environment.

## Automark assessment

The automarker reads a CSV log from `logs/lab4.csv` (override with the `LAB4_LOG` environment variable).  
See `tests/test_automark_metrics.py` (`test_lab4_automark_metrics_from_log`) for the full evaluation logic.

### Required CSV columns

| Column | Description |
| --- | --- |
| `run_id` | Unique identifier for each navigation run |
| `success` | Whether the run succeeded (`true`/`false`) |
| `distance_remaining` | Distance remaining to goal at end of run (m) |

### Evaluated metrics and pass thresholds

| Metric | Default threshold | Override env variable |
| --- | --- | --- |
| Success rate | ≥ 0.80 | `LAB4_MIN_SUCCESS_RATE` |
| Mean distance remaining for unsuccessful runs | ≤ 0.40 m | `LAB4_MAX_UNSUCCESSFUL_DISTANCE_M` |

Setting yourself up for the final demo:
-
* The final demo requires racing another robot in an unknown environment.
* You should prepare a robust integrated pipeline that can generalise beyond known tracks.
* Consider combining planning, mapping, and control modules, or training an RL-based approach for robust behaviour.
