# Phase 2 Retrieval Scale Report

Decision: **NOT_READY_FOR_HPO**

| Dataset | Model | Stage 1 | Total (s) | Train (s) | Val (s) | Metric | Peak CUDA GiB | Candidates | Query throughput (q/s) | Stage 2 | Trial time (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| adult | tabr | PASS | 10.6189 | 5.8491 | 0.5679 | 0.8348 | 0.5598 | 39073 | 445426.0874 | PASS | 163.9765 |
| adult | modernnca | PASS | 8.0658 | 3.0891 | 0.9297 | 0.8440 | 0.4283 | 39073 | 48121.7544 | PASS | 155.1236 |
| covertype | tabr | PASS | 427.9725 | 373.4627 | 43.5090 | 0.9566 | 0.9375 | 464809 | 57670.0220 | TIMEOUT | >=900.2740 |
| covertype | modernnca | PASS | 254.2669 | 193.1042 | 49.4024 | 0.9720 | 7.5847 | 464809 | 8545.0781 | TIMEOUT | >=900.2029 |

## Blockers

- covertype/tabr stage2 TIMEOUT: Exceeded 900 seconds
- covertype/modernnca stage2 TIMEOUT: Exceeded 900 seconds

## Cost estimate

```json
{
  "basis": "completed trial wall time; timeout durations are lower bounds; 9-dataset estimate uses the Adult/Covertype mean per model",
  "per_combination": [
    {
      "dataset": "adult",
      "model": "tabr",
      "observed_trial_seconds": 163.97654048725963,
      "is_lower_bound": false,
      "fifty_trial_gpu_hours": 2.277451951211939
    },
    {
      "dataset": "adult",
      "model": "modernnca",
      "observed_trial_seconds": 155.12361677363515,
      "is_lower_bound": false,
      "fifty_trial_gpu_hours": 2.1544946774115994
    },
    {
      "dataset": "covertype",
      "model": "tabr",
      "observed_trial_seconds": 900.273988660425,
      "is_lower_bound": true,
      "fifty_trial_gpu_hours": 12.503805398061457
    },
    {
      "dataset": "covertype",
      "model": "modernnca",
      "observed_trial_seconds": 900.2028951924294,
      "is_lower_bound": true,
      "fifty_trial_gpu_hours": 12.502817988783741
    }
  ],
  "observed_matrix_fifty_trial_gpu_hours_lower_bound": 29.438570015468738,
  "nine_dataset_two_model_gpu_hours_lower_bound": 132.47356506960932,
  "available_gpu_count": 3,
  "ideal_wall_clock_hours_at_available_gpu_count_lower_bound": 44.157855023203105,
  "estimated_hpo_disk_bytes": 90886878.0,
  "disk_estimate_basis": "50 JSON artifacts plus one retained checkpoint per dataset/model; excludes SQLite and log overhead"
}
```

Timeout-derived costs are lower bounds, not completion estimates.

No production 50-trial HPO or 15-seed final evaluation was launched.
