# NCAA Basketball Point Spreads — Triangle Sports Analytics 2026

Winning solo entry (**devig to ball knowledge**, UNC) in the [2026 Triangle Sports Analytics competition](https://triangle-sports.github.io/): **1st place in point spreads** and **1st place in prediction intervals** out of 41 student teams from UNC, Duke, and NC State.

The task was to forecast home-minus-away margins for 78 ACC men's basketball games (Feb 7–Mar 7, 2026) from a fixed pre-window cutoff, then attach intervals that stayed valid under a 70% coverage rule.

| Track | Result | Score |
| --- | --- | --- |
| Point spreads | 1st / 41 | MAE **8.629** |
| Prediction intervals | 1st | **70.51%** coverage, PIW **22.45** |

Official standings: [winners](https://triangle-sports.github.io/winners.html) · [leaderboard](https://triangle-sports.github.io/)

## Approach

The pipeline is a **ridge regression baseline** with a **shallow XGBoost residual correction**, trained on Division I games and applied to ACC matchups.

### Features (no leakage)

Public team-level efficiency and context, expressed as home/away differences:

- Bart Torvik adjusted offense/defense, tempo, and related Four Factors-style stats
- Rest and congestion from the full D1 schedule (days since last game, games in the last 7 days)
- Free-throw rate / foul-pressure interactions
- Neutral-site indicator

Ratings are merged on `as_of_date = game_date - 1 day`. Walk-forward folds freeze every team's snapshot at the last date before the test window, so “future” games cannot see in-window updates.

### Point model

1. **Ridge** on a small production feature set (efficiency diffs, tempo × efficiency, fatigue asymmetry, game tempo, free-throw leverage), with `StandardScaler` and `alpha` chosen by inner `TimeSeriesSplit`.
2. **XGBoost** is fit with the ridge prediction as `base_margin`, so the trees learn the residual rather than the full spread. The booster is heavily regularized (`max_depth=2`, large `min_child_weight` / `reg_lambda`) to avoid chasing noise in a short season.
3. **Locked market shrink (optional).** If a projection or close exists, the submitted point is a blend with weights chosen on earlier walk-forward folds, not on contest games: 70% model / 30% EvanMiya, then 50/50 with the ESPN close. If either source is missing, the pipeline falls back to the unblended ridge+XGB prediction.

### Prediction intervals

Competition scoring: intervals are **disqualified below 70% coverage**; among valid entries, rank by average width (PIW).

Widths come from **empirical residual CDFs** stratified by predicted spread, tempo, and offense–defense imbalance. A greedy allocator adds width where it buys the most coverage, then a calibration multiplier is tuned on a held-out time window to sit just above the 70% bar. Interval **centers** can follow the submitted point (including the locked blend); **widths** are still fit from model residuals.

## Repository

```
models/     ridge, XGBoost, and interval logic
scripts/    data collection, evaluation, and submission runners
data/       local CSVs (gitignored; scrape with scripts below)
```

| Path | Role |
| --- | --- |
| `scripts/data_collection.py` | ESPN schedules, scores, and closing lines |
| `scripts/scrape_torvik.py` | Bart Torvik ratings mapped to ESPN team IDs |
| `models/ridge_model.py` | Features, as-of joins, walk-forward ridge |
| `models/xgboost_model.py` | Ridge → XGBoost residual model |
| `models/prediction_intervals.py` | Interval backtest and forecasts |
| `scripts/xgb_final_point_submission_run.py` | Fold-locked Miya/Vegas blend (`sync_evanmiya_priority.py`, `xgb_miya_blend_w_tune.py`) |

Run from the repo root, e.g. `python scripts/data_collection.py`.

**Stack:** Python, pandas, NumPy, scikit-learn, XGBoost.
