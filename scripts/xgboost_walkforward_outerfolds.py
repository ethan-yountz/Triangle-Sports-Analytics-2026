"""Walk-forward XGBoost evaluation on OUTER_FOLDS with snapshot locking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from models.ridge_model import (
    OUTER_FOLDS,
    RIDGE_FINAL_FEATURES,
    SNAPSHOT_STATS,
    add_engineered_features,
    apply_fixed_cutoff_snapshot,
    build_base_model_frame,
    choose_tscv_splits,
    lock_non_torvik_features_to_fold_cutoff,
)


RIDGE_STACK_FEATURES = list(RIDGE_FINAL_FEATURES)

BLOCK_6_FEATURES = [
    "tempo_x_adj_o_diff",
    "tempo_x_adj_d_diff",
    "tempo_x_efg_diff",
    "tempo_x_efgd_diff",
]
BLOCK_9_FEATURES = [
    "days_since_last_game_diff",
    "games_last_7_days_diff",
    "fatigue_index_diff",
]
BLOCK_10_FEATURES = ["adj_tempo_avg"]
BLOCK_11_FEATURES = [
    "ftr_diff",
    "ftr_allowed_diff",
    "foul_rate_diff",
    "opponent_foul_rate_diff",
    "ftr_x_ridge_pred",
    "ftr_x_fatigue",
]

XGB_FEATURES = [
    "adj_o_diff",
    "adj_d_diff",
    *BLOCK_6_FEATURES,
    *BLOCK_9_FEATURES,
    *BLOCK_10_FEATURES,
    *BLOCK_11_FEATURES,
]
OUTPUT_PATH = os.path.join("data", "xgboost_walkforward_outerfold_results.csv")
FINAL_PRED_SHIFT = -1.0


def add_ridge_stack_feature(
    df: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=["margin"] + RIDGE_STACK_FEATURES).copy()

    train_mask = train_mask.reindex(out.index).fillna(False).astype(bool)
    test_mask = test_mask.reindex(out.index).fillna(False).astype(bool)
    if int(train_mask.sum()) < 100:
        raise ValueError("Insufficient train rows for ridge stack feature.")
    if int(test_mask.sum()) < 5:
        raise ValueError("Insufficient test rows for ridge stack feature.")

    X_train = out.loc[train_mask, RIDGE_STACK_FEATURES].values
    y_train = pd.to_numeric(out.loc[train_mask, "margin"], errors="coerce").values
    X_all = out[RIDGE_STACK_FEATURES].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_all_s = scaler.transform(X_all)

    n_splits = min(choose_tscv_splits(len(X_train_s)), len(X_train_s) - 1)
    if n_splits < 2:
        n_splits = 2
    ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=TimeSeriesSplit(n_splits=n_splits))
    ridge_cv.fit(X_train_s, y_train)
    ridge = Ridge(alpha=float(ridge_cv.alpha_))
    ridge.fit(X_train_s, y_train)

    out["ridge_pred_spread"] = ridge.predict(X_all_s)
    out["ftr_x_ridge_pred"] = pd.to_numeric(out["ftr_diff"], errors="coerce") * np.abs(
        pd.to_numeric(out["ridge_pred_spread"], errors="coerce")
    )
    return out


def fit_xgb_fold(
    df: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    test_start: pd.Timestamp,
) -> Dict[str, object]:
    work = df.dropna(subset=["margin", "ridge_pred_spread"] + XGB_FEATURES).copy()
    train_mask = train_mask.reindex(work.index).fillna(False).astype(bool)
    test_mask = test_mask.reindex(work.index).fillna(False).astype(bool)

    train_df = work.loc[train_mask].copy()
    test_df = work.loc[test_mask].copy()

    X_test = test_df[XGB_FEATURES].values
    y_test = pd.to_numeric(test_df["margin"], errors="coerce").values
    base_test = pd.to_numeric(test_df["ridge_pred_spread"], errors="coerce").values

    if len(train_df) < 150:
        raise ValueError("Insufficient XGB train rows.")
    if len(X_test) < 5:
        raise ValueError("Insufficient XGB test rows.")

    train_dates = pd.to_datetime(train_df["date_dt"], errors="coerce")
    if train_dates.isna().all():
        raise ValueError("Missing train dates for validation split.")

    val_start = test_start - pd.Timedelta(days=14)
    train_inner_mask = train_dates < val_start
    val_inner_mask = (train_dates >= val_start) & (train_dates < test_start)

    inner_train_df = train_df.loc[train_inner_mask].copy()
    inner_val_df = train_df.loc[val_inner_mask].copy()

    X_inner_train = inner_train_df[XGB_FEATURES].values
    y_inner_train = pd.to_numeric(inner_train_df["margin"], errors="coerce").values
    base_inner_train = pd.to_numeric(inner_train_df["ridge_pred_spread"], errors="coerce").values

    X_inner_val = inner_val_df[XGB_FEATURES].values
    y_inner_val = pd.to_numeric(inner_val_df["margin"], errors="coerce").values
    base_inner_val = pd.to_numeric(inner_val_df["ridge_pred_spread"], errors="coerce").values

    base_params = dict(
        objective="reg:pseudohubererror",
        max_depth=2,
        min_child_weight=80,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.25,
        reg_alpha=0.0,
        reg_lambda=120.0,
        random_state=42,
        eval_metric="mae",
    )

    xgb = XGBRegressor(
        **base_params,
        n_estimators=4000,
        early_stopping_rounds=200,
    )

    can_do_early_stop = len(X_inner_train) >= 50 and len(X_inner_val) >= 10
    if not can_do_early_stop:
        raise ValueError("Insufficient inner split rows for fold-safe training.")

    xgb.fit(
        X_inner_train,
        y_inner_train,
        base_margin=base_inner_train,
        eval_set=[(X_inner_val, y_inner_val)],
        base_margin_eval_set=[base_inner_val],
        verbose=False,
    )

    best_iter_raw = getattr(xgb, "best_iteration", None)
    best_iter = int(best_iter_raw) if best_iter_raw is not None else 3999
    pred_val_xgb = xgb.predict(
        X_inner_val,
        base_margin=base_inner_val,
        iteration_range=(0, best_iter + 1),
    )

    # Refit on full outer-train with fixed tree count from early-stop best_iter.
    X_outer_train = train_df[XGB_FEATURES].values
    y_outer_train = pd.to_numeric(train_df["margin"], errors="coerce").values
    base_outer_train = pd.to_numeric(train_df["ridge_pred_spread"], errors="coerce").values
    xgb_refit = XGBRegressor(
        **base_params,
        n_estimators=best_iter + 1,
    )
    xgb_refit.fit(X_outer_train, y_outer_train, base_margin=base_outer_train, verbose=False)
    y_pred_raw = xgb_refit.predict(X_test, base_margin=base_test)
    y_pred = y_pred_raw + FINAL_PRED_SHIFT

    abs_error = np.abs(y_test - y_pred)
    mae = float(mean_absolute_error(y_test, y_pred))

    spread = pd.to_numeric(test_df.get("spread"), errors="coerce")
    vegas_mask = spread.notna()
    vegas_n = int(vegas_mask.sum())
    if vegas_n > 0:
        vegas_pred = -spread.loc[vegas_mask].to_numpy(dtype=float)
        y_vegas_true = y_test[vegas_mask.to_numpy()]
        y_vegas_model = y_pred[vegas_mask.to_numpy()]
        vegas_mae = float(mean_absolute_error(y_vegas_true, vegas_pred))
        model_mae_on_vegas_rows = float(mean_absolute_error(y_vegas_true, y_vegas_model))
    else:
        vegas_mae = np.nan
        model_mae_on_vegas_rows = np.nan

    return {
        "mae": mae,
        "test_n": int(len(y_test)),
        "abs_error": abs_error,
        "vegas_n": vegas_n,
        "vegas_mae": vegas_mae,
        "model_mae_on_vegas_rows": model_mae_on_vegas_rows,
        "pred_shift_applied": float(FINAL_PRED_SHIFT),
        "best_iteration": int(best_iter),
        "inner_train_n": int(len(X_inner_train)),
        "inner_val_n": int(len(X_inner_val)),
        "inner_val_start": val_start.date().isoformat(),
        "used_early_stopping": bool(can_do_early_stop),
        "inner_val_mae_base": float(mean_absolute_error(y_inner_val, base_inner_val)),
        "inner_val_mae_xgb": float(mean_absolute_error(y_inner_val, pred_val_xgb)),
    }


def main() -> None:
    base_df, torvik = build_base_model_frame()
    fold_rows: List[Dict[str, object]] = []
    pooled_abs: List[np.ndarray] = []

    print("XGBoost walk-forward on OUTER_FOLDS")
    print(f"xgb_features: {XGB_FEATURES}")

    for fold in OUTER_FOLDS:
        fold_df = apply_fixed_cutoff_snapshot(base_df, torvik, fold.snapshot_cutoff, SNAPSHOT_STATS)
        fold_df = lock_non_torvik_features_to_fold_cutoff(fold_df, fold.snapshot_cutoff)
        fold_df = add_engineered_features(fold_df).sort_values("date_dt").reset_index(drop=True)

        train_mask = fold_df["date_dt"] < fold.test_start
        test_mask = (fold_df["date_dt"] >= fold.test_start) & (fold_df["date_dt"] < fold.test_end)
        train_n_raw = int(train_mask.sum())
        test_n_raw = int(test_mask.sum())
        if int(test_mask.sum()) < 5:
            print(f"{fold.name} skipped: no test rows.")
            continue

        fold_df = add_ridge_stack_feature(fold_df, train_mask, test_mask)
        missing = [c for c in XGB_FEATURES if c not in fold_df.columns]
        if missing:
            raise ValueError(f"Missing XGB features: {missing}")

        print(
            f"\n=== {fold.name} | train < {fold.test_start.date()} | "
            f"test [{fold.test_start.date()}, {fold.test_end.date()}) ==="
        )
        print(f"snapshot_cutoff_date_for_test_features: {fold.snapshot_cutoff.date()}")
        print(f"train_n_raw: {train_n_raw}, test_n_raw: {test_n_raw}")

        result = fit_xgb_fold(fold_df, train_mask, test_mask, fold.test_start)

        pooled_abs.append(np.asarray(result["abs_error"], dtype=float))
        fold_rows.append(
            {
                "fold": fold.name,
                "train_end_exclusive": fold.test_start.date().isoformat(),
                "test_start_inclusive": fold.test_start.date().isoformat(),
                "test_end_exclusive": fold.test_end.date().isoformat(),
                "snapshot_cutoff_date": fold.snapshot_cutoff.date().isoformat(),
                "test_n": int(result["test_n"]),
                "test_mae": float(result["mae"]),
                "vegas_n": int(result["vegas_n"]),
                "vegas_mae": float(result["vegas_mae"]) if np.isfinite(result["vegas_mae"]) else np.nan,
                "model_mae_on_vegas_rows": float(result["model_mae_on_vegas_rows"])
                if np.isfinite(result["model_mae_on_vegas_rows"])
                else np.nan,
                "pred_shift_applied": float(result["pred_shift_applied"]),
                "best_iteration": int(result["best_iteration"]),
                "inner_train_n": int(result["inner_train_n"]),
                "inner_val_n": int(result["inner_val_n"]),
                "inner_val_start": str(result["inner_val_start"]),
                "used_early_stopping": bool(result["used_early_stopping"]),
                "inner_val_mae_base": float(result["inner_val_mae_base"]),
                "inner_val_mae_xgb": float(result["inner_val_mae_xgb"]),
            }
        )
        print(
            f"xgb_test_mae: {result['mae']:.4f} | test_n={result['test_n']} | "
            f"model_mae_vegas_rows={result['model_mae_on_vegas_rows']:.4f} | "
            f"vegas_mae={result['vegas_mae']:.4f} | vegas_n={result['vegas_n']} | "
            f"best_iteration={result['best_iteration']} | "
            f"inner14_train_n={result['inner_train_n']} | "
            f"inner14_val_n={result['inner_val_n']} | "
            f"val_mae_base={result['inner_val_mae_base']:.4f} | "
            f"val_mae_xgb={result['inner_val_mae_xgb']:.4f} | "
            f"used_early_stopping={result['used_early_stopping']}"
        )

    if len(pooled_abs) == 0:
        raise RuntimeError("No fold results were produced.")

    pooled_mae = float(np.mean(np.concatenate(pooled_abs)))
    fold_df_out = pd.DataFrame(fold_rows)
    fold_df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"\npooled_test_rows: {int(fold_df_out['test_n'].sum())}")
    print(f"pooled_mae: {pooled_mae:.6f}")
    if fold_df_out["vegas_n"].sum() > 0:
        weighted_model_vs_vegas = float(
            np.nansum(
                pd.to_numeric(fold_df_out["model_mae_on_vegas_rows"], errors="coerce")
                * pd.to_numeric(fold_df_out["vegas_n"], errors="coerce")
            )
            / max(float(pd.to_numeric(fold_df_out["vegas_n"], errors="coerce").sum()), 1.0)
        )
        weighted_vegas = float(
            np.nansum(
                pd.to_numeric(fold_df_out["vegas_mae"], errors="coerce")
                * pd.to_numeric(fold_df_out["vegas_n"], errors="coerce")
            )
            / max(float(pd.to_numeric(fold_df_out["vegas_n"], errors="coerce").sum()), 1.0)
        )
        print(f"pooled_model_mae_on_vegas_rows: {weighted_model_vs_vegas:.6f}")
        print(f"pooled_vegas_mae: {weighted_vegas:.6f}")
    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
