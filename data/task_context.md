# Triangle Sports Task Context

Source URL: https://triangle-sports.github.io/task.html
Saved on: 2026-02-06

Key requirements captured from the page:
- Objective (point spread): predict point spread (home - away) for 78 ACC games.
- Evaluation metric for point spread: MAE over all 78 games.
- Optional objective (prediction intervals): submit lower/upper bounds for all 78 games.
- PI interval rule: coverage must be at least 70%; otherwise disqualified.
- If coverage >= 70%, interval score is average interval width (smaller is better).
- Submission guidance emphasizes preserving first columns/order in template and filling all required NA predictions.

Modeling implication saved for this repo context:
- Backtests should use Torvik features that would have been available before each game.
- For training/testing in this repo, use prior-day Torvik snapshot (as_of_date = game_date - 1 day) to avoid same-day leakage.
