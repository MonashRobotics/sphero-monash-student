# Project: Race

The project is a race against another robot in an unknown environment. You should prepare a robust integrated pipeline that combines your planning, mapping, and control modules, or train an RL-based approach for robust behaviour.

## Key learning outcomes:
- Integrate all modules from Labs 1–4 into a fully autonomous racing pipeline
- Generalise your solution beyond known environments
- Compete against other teams and track relative ranking performance

## Automark assessment

The automarker reads a CSV log from `logs/project_race.csv` (override with the `PROJECT_LOG` environment variable).  
See `tests/test_automark_metrics.py` (`test_project_automark_metrics_from_log`) for the full evaluation logic.

### Required CSV columns

| Column | Description |
| --- | --- |
| `run_id` | Unique identifier for each race run |
| `success` | Whether the run completed successfully (`true`/`false`) |
| `distance_remaining` | Distance remaining to goal at end of run (m) |
| `time_to_goal_s` | Time taken to reach the goal for successful runs (s) |
| `class_rank` | Your team's rank in the class for this run |
| `class_size` | Total number of teams competing |

### Evaluated metrics and pass thresholds

| Metric | Default threshold | Override env variable |
| --- | --- | --- |
| Success rate | ≥ 0.70 | `PROJECT_MIN_SUCCESS_RATE` |
| Median time to goal (successful runs) | ≤ 120.0 s | `PROJECT_MAX_MEDIAN_TIME_TO_GOAL_S` |
| Mean distance remaining for unsuccessful runs | ≤ 0.50 m | `PROJECT_MAX_UNSUCCESSFUL_DISTANCE_M` |
| Best rank percentile (`min(class_rank / class_size)`) | ≤ 0.50 | `PROJECT_MAX_RANK_PERCENTILE` |
