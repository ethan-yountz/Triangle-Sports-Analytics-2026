"""Tune XGB/Miya blend weight on fixed walk-forward outer folds."""

import argparse
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


OOF_PATH = os.path.join("data", "xgb_oof_predictions_with_gameids.csv")
GAMES_PATH = os.path.join("data", "all_games.csv")
SWEEP_OUT = os.path.join("data", "xgb_miya_blend_w_sweep.csv")
SUMMARY_OUT = os.path.join("data", "xgb_miya_blend_w_summary.csv")
FOLD_METRICS_OUT = os.path.join("data", "xgb_miya_blend_fold_metrics.csv")

TRAIN_FOLDS = [f"Fold {i}" for i in range(1, 7)]
HOLDOUT_FOLD = "Fold 7"
DEFAULT_W_GRID = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
INTERCEPT_SHIFT = 1.0
MIYA_AGE_COL_CANDIDATES = ["miya_age_days", "evan_age_days", "evan_spread_age_days"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune blend weight w for two XGB baselines against Miya proxy (evan_spread), "
            "using folds 1-6 only, then evaluate on fold 7."
        )
    )
    parser.add_argument(
        "--oof-path",
        default=OOF_PATH,
        help="Path to OOF predictions with fold/event_id/pred_raw/actual_margin.",
    )
    parser.add_argument(
        "--games-path",
        default=GAMES_PATH,
        help="Path to all_games CSV containing event_id and evan_spread.",
    )
    parser.add_argument(
        "--w-grid",
        default=",".join(f"{w:.2f}" for w in DEFAULT_W_GRID),
        help="Comma-separated candidate weights for XGB contribution.",
    )
    parser.add_argument(
        "--max-miya-age-days",
        type=float,
        default=None,
        help=(
            "Optional stale cutoff for Miya rows if an age column exists "
            f"({', '.join(MIYA_AGE_COL_CANDIDATES)})."
        ),
    )
    parser.add_argument("--sweep-out", default=SWEEP_OUT, help="Output CSV path for full w sweep.")
    parser.add_argument("--summary-out", default=SUMMARY_OUT, help="Output CSV path for selected weights.")
    parser.add_argument(
        "--fold-metrics-out",
        default=FOLD_METRICS_OUT,
        help="Output CSV path for per-fold MAE diagnostics at selected weights.",
    )
    return parser.parse_args()


def parse_w_grid(raw: str) -> List[float]:
    weights: List[float] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        val = float(text)
        if val < 0.0 or val > 1.0:
            raise ValueError(f"w must be in [0,1], got {val}.")
        weights.append(val)
    if not weights:
        raise ValueError("No candidate w values parsed from --w-grid.")
    return weights


def fold_sort_key(name: str) -> Tuple[int, str]:
    text = str(name)
    parts = text.split()
    if len(parts) == 2 and parts[0] == "Fold":
        try:
            return int(parts[1]), text
        except ValueError:
            pass
    return 999, text


def load_join_frame(oof_path: str, games_path: str) -> pd.DataFrame:
    oof = pd.read_csv(oof_path)
    required_oof_cols = {"fold", "event_id", "pred_raw", "actual_margin"}
    missing_oof = sorted(required_oof_cols - set(oof.columns))
    if missing_oof:
        raise ValueError(f"OOF file missing required columns: {missing_oof}")

    games_header = pd.read_csv(games_path, nrows=0).columns.tolist()
    usecols = ["event_id", "evan_spread"]
    for c in MIYA_AGE_COL_CANDIDATES:
        if c in games_header:
            usecols.append(c)
    games = pd.read_csv(games_path, usecols=usecols)

    oof["event_id"] = pd.to_numeric(oof["event_id"], errors="coerce").astype("Int64")
    games["event_id"] = pd.to_numeric(games["event_id"], errors="coerce").astype("Int64")
    oof["pred_raw"] = pd.to_numeric(oof["pred_raw"], errors="coerce")
    oof["actual_margin"] = pd.to_numeric(oof["actual_margin"], errors="coerce")
    games["evan_spread"] = pd.to_numeric(games["evan_spread"], errors="coerce")

    out = oof.merge(games, on="event_id", how="left", validate="many_to_one")
    out = out.dropna(subset=["pred_raw", "actual_margin"]).copy()

    # evan_spread is home spread; home-margin orientation is -spread.
    out["miya_raw"] = -pd.to_numeric(out["evan_spread"], errors="coerce")
    out["xgb_base"] = pd.to_numeric(out["pred_raw"], errors="coerce") - INTERCEPT_SHIFT
    out["xgb_no"] = pd.to_numeric(out["pred_raw"], errors="coerce")
    return out


def build_miya_usable_mask(
    df: pd.DataFrame, max_miya_age_days: Optional[float]
) -> Tuple[pd.Series, Optional[str], int]:
    usable = df["miya_raw"].notna().copy()
    if max_miya_age_days is None:
        return usable, None, 0

    for col in MIYA_AGE_COL_CANDIDATES:
        if col not in df.columns:
            continue
        ages = pd.to_numeric(df[col], errors="coerce")
        stale = ages > float(max_miya_age_days)
        stale_n = int((usable & stale).sum())
        usable = usable & (~stale)
        return usable, col, stale_n

    return usable, None, 0


def blend_with_fallback(
    xgb_pred: np.ndarray,
    miya_pred: np.ndarray,
    miya_usable: np.ndarray,
    w: float,
) -> np.ndarray:
    out = np.asarray(xgb_pred, dtype=float).copy()
    miya = np.asarray(miya_pred, dtype=float)
    usable = np.asarray(miya_usable, dtype=bool)
    out[usable] = w * out[usable] + (1.0 - w) * miya[usable]
    return out


def run_sweep(df: pd.DataFrame, w_grid: Sequence[float]) -> pd.DataFrame:
    train_mask = df["fold"].isin(TRAIN_FOLDS)
    holdout_mask = df["fold"] == HOLDOUT_FOLD
    if int(train_mask.sum()) == 0:
        raise RuntimeError("No rows in folds 1-6.")
    if int(holdout_mask.sum()) == 0:
        raise RuntimeError("No rows in Fold 7.")

    actual = pd.to_numeric(df["actual_margin"], errors="coerce").to_numpy(dtype=float)
    xgb_base = pd.to_numeric(df["xgb_base"], errors="coerce").to_numpy(dtype=float)
    xgb_no = pd.to_numeric(df["xgb_no"], errors="coerce").to_numpy(dtype=float)
    miya_raw = pd.to_numeric(df["miya_raw"], errors="coerce").to_numpy(dtype=float)
    miya_usable = df["miya_usable"].to_numpy(dtype=bool)
    train_idx = train_mask.to_numpy(dtype=bool)
    holdout_idx = holdout_mask.to_numpy(dtype=bool)

    rows = []
    for w in w_grid:
        pred_base = blend_with_fallback(xgb_base, miya_raw, miya_usable, float(w))
        pred_no = blend_with_fallback(xgb_no, miya_raw, miya_usable, float(w))

        rows.append(
            {
                "w": float(w),
                "pooled_mae_folds1_6_base": float(
                    mean_absolute_error(actual[train_idx], pred_base[train_idx])
                ),
                "pooled_mae_folds1_6_no": float(
                    mean_absolute_error(actual[train_idx], pred_no[train_idx])
                ),
                "fold7_mae_base": float(mean_absolute_error(actual[holdout_idx], pred_base[holdout_idx])),
                "fold7_mae_no": float(mean_absolute_error(actual[holdout_idx], pred_no[holdout_idx])),
            }
        )

    return pd.DataFrame(rows)


def build_fold_metrics(df: pd.DataFrame, w_base: float, w_no: float) -> pd.DataFrame:
    pred_base = blend_with_fallback(
        df["xgb_base"].to_numpy(dtype=float),
        df["miya_raw"].to_numpy(dtype=float),
        df["miya_usable"].to_numpy(dtype=bool),
        w_base,
    )
    pred_no = blend_with_fallback(
        df["xgb_no"].to_numpy(dtype=float),
        df["miya_raw"].to_numpy(dtype=float),
        df["miya_usable"].to_numpy(dtype=bool),
        w_no,
    )

    work = df.copy()
    work["pred_base_wstar"] = pred_base
    work["pred_no_wstar"] = pred_no

    rows = []
    for fold_name, grp in sorted(work.groupby("fold"), key=lambda x: fold_sort_key(str(x[0]))):
        y = pd.to_numeric(grp["actual_margin"], errors="coerce").to_numpy(dtype=float)
        rows.append(
            {
                "fold": str(fold_name),
                "n": int(len(grp)),
                "miya_usable_n": int(grp["miya_usable"].sum()),
                "miya_usable_rate": float(grp["miya_usable"].mean()),
                "mae_xgb_base_w1": float(
                    mean_absolute_error(y, pd.to_numeric(grp["xgb_base"], errors="coerce").to_numpy(dtype=float))
                ),
                "mae_xgb_no_w1": float(
                    mean_absolute_error(y, pd.to_numeric(grp["xgb_no"], errors="coerce").to_numpy(dtype=float))
                ),
                "mae_pred_base_wstar": float(
                    mean_absolute_error(y, pd.to_numeric(grp["pred_base_wstar"], errors="coerce").to_numpy(dtype=float))
                ),
                "mae_pred_no_wstar": float(
                    mean_absolute_error(y, pd.to_numeric(grp["pred_no_wstar"], errors="coerce").to_numpy(dtype=float))
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    w_grid = parse_w_grid(args.w_grid)
    df = load_join_frame(args.oof_path, args.games_path)
    miya_usable, stale_col, stale_n = build_miya_usable_mask(df, args.max_miya_age_days)
    df["miya_usable"] = miya_usable

    sweep_df = run_sweep(df, w_grid)
    base_best_idx = int(sweep_df["pooled_mae_folds1_6_base"].idxmin())
    no_best_idx = int(sweep_df["pooled_mae_folds1_6_no"].idxmin())
    w_base = float(sweep_df.loc[base_best_idx, "w"])
    w_no = float(sweep_df.loc[no_best_idx, "w"])

    summary_df = pd.DataFrame(
        [
            {
                "variant": "base_intercept_corrected",
                "selected_w": w_base,
                "pooled_mae_folds1_6": float(sweep_df.loc[base_best_idx, "pooled_mae_folds1_6_base"]),
                "fold7_mae": float(sweep_df.loc[base_best_idx, "fold7_mae_base"]),
                "train_rows": int(df["fold"].isin(TRAIN_FOLDS).sum()),
                "fold7_rows": int((df["fold"] == HOLDOUT_FOLD).sum()),
                "miya_usable_rate_train": float(df.loc[df["fold"].isin(TRAIN_FOLDS), "miya_usable"].mean()),
                "miya_usable_rate_fold7": float(df.loc[df["fold"] == HOLDOUT_FOLD, "miya_usable"].mean()),
                "staleness_col_used": stale_col or "",
                "max_miya_age_days": args.max_miya_age_days if args.max_miya_age_days is not None else np.nan,
                "miya_stale_rows_excluded": stale_n,
            },
            {
                "variant": "no_intercept_raw",
                "selected_w": w_no,
                "pooled_mae_folds1_6": float(sweep_df.loc[no_best_idx, "pooled_mae_folds1_6_no"]),
                "fold7_mae": float(sweep_df.loc[no_best_idx, "fold7_mae_no"]),
                "train_rows": int(df["fold"].isin(TRAIN_FOLDS).sum()),
                "fold7_rows": int((df["fold"] == HOLDOUT_FOLD).sum()),
                "miya_usable_rate_train": float(df.loc[df["fold"].isin(TRAIN_FOLDS), "miya_usable"].mean()),
                "miya_usable_rate_fold7": float(df.loc[df["fold"] == HOLDOUT_FOLD, "miya_usable"].mean()),
                "staleness_col_used": stale_col or "",
                "max_miya_age_days": args.max_miya_age_days if args.max_miya_age_days is not None else np.nan,
                "miya_stale_rows_excluded": stale_n,
            },
        ]
    )
    fold_metrics_df = build_fold_metrics(df, w_base=w_base, w_no=w_no)

    sweep_df.to_csv(args.sweep_out, index=False)
    summary_df.to_csv(args.summary_out, index=False)
    fold_metrics_df.to_csv(args.fold_metrics_out, index=False)

    print("Miya blend weight tuning complete.")
    print(f"oof_path: {args.oof_path}")
    print(f"games_path: {args.games_path}")
    print(f"rows_total: {len(df)}")
    print(f"rows_train_folds_1_6: {int(df['fold'].isin(TRAIN_FOLDS).sum())}")
    print(f"rows_fold7: {int((df['fold'] == HOLDOUT_FOLD).sum())}")
    print(f"miya_usable_rate_all: {float(df['miya_usable'].mean()):.6f}")
    if args.max_miya_age_days is not None and stale_col is None:
        print("warning: max_miya_age_days set, but no Miya age column found; stale filtering not applied.")
    if stale_col is not None:
        print(f"staleness_col_used: {stale_col}")
        print(f"miya_stale_rows_excluded: {stale_n}")
    print("\nSweep grid:")
    print(sweep_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSelected:")
    print(
        summary_df[
            [
                "variant",
                "selected_w",
                "pooled_mae_folds1_6",
                "fold7_mae",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
    print(f"\nsaved: {args.sweep_out}")
    print(f"saved: {args.summary_out}")
    print(f"saved: {args.fold_metrics_out}")


if __name__ == "__main__":
    main()
