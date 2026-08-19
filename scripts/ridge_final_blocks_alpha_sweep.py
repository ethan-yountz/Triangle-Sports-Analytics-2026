"""Alpha sweep for fixed final block set, then all-fold model vs Vegas MAE."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.ridge_model import (
    ALPHA_GRID,
    FEATURE_BLOCKS,
    OUTER_FOLDS,
    SNAPSHOT_STATS,
    add_engineered_features,
    apply_fixed_cutoff_snapshot,
    build_base_model_frame,
    lock_non_torvik_features_to_fold_cutoff,
)


FINAL_BLOCKS = [6, 9, 10, 11]
BASELINE_FEATURES = ["adj_o_diff", "adj_d_diff", "is_neutral"]
MIN_TRAIN_ROWS = 50
MIN_TEST_ROWS = 5

ALPHA_SWEEP_OUT = os.path.join("data", "final_blocks_alpha_sweep.csv")
FOLD_RESULTS_OUT = os.path.join("data", "final_blocks_fold_results.csv")


def unique_in_order(values: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def block_id_to_features() -> Dict[int, List[str]]:
    mapping: Dict[int, List[str]] = {}
    for idx, (_, feats) in enumerate(FEATURE_BLOCKS, start=1):
        mapping[idx] = list(feats)
    missing = [b for b in FINAL_BLOCKS if b not in mapping]
    if missing:
        raise ValueError(f"Missing FINAL_BLOCKS in FEATURE_BLOCKS: {missing}")
    return mapping


def build_final_feature_list() -> List[str]:
    mapping = block_id_to_features()
    feats = list(BASELINE_FEATURES)
    for block_id in FINAL_BLOCKS:
        feats.extend(mapping[block_id])
    return unique_in_order(feats)


def preprocess_fold_df(base_df: pd.DataFrame, torvik: pd.DataFrame, fold) -> pd.DataFrame:
    out = apply_fixed_cutoff_snapshot(base_df, torvik, fold.snapshot_cutoff, SNAPSHOT_STATS)
    out = lock_non_torvik_features_to_fold_cutoff(out, fold.snapshot_cutoff)
    out = add_engineered_features(out).sort_values("date_dt").reset_index(drop=True)
    return out


def fit_fixed_alpha_on_fold(
    fold_df: pd.DataFrame,
    fold,
    feature_cols: Sequence[str],
    alpha: float,
) -> Dict[str, object]:
    work = fold_df.copy()
    work["margin"] = pd.to_numeric(work["margin"], errors="coerce")
    work = work.dropna(subset=["margin"]).copy()

    train_mask = work["date_dt"] < fold.test_start
    test_mask = (work["date_dt"] >= fold.test_start) & (work["date_dt"] < fold.test_end)
    train_n = int(train_mask.sum())
    test_n = int(test_mask.sum())
    if train_n < MIN_TRAIN_ROWS:
        raise ValueError(f"{fold.name}: insufficient train rows ({train_n}).")
    if test_n < MIN_TEST_ROWS:
        raise ValueError(f"{fold.name}: insufficient test rows ({test_n}).")

    X_train = work.loc[train_mask, list(feature_cols)]
    X_test = work.loc[test_mask, list(feature_cols)]
    y_train = work.loc[train_mask, "margin"].to_numpy(dtype=float)
    y_test = work.loc[test_mask, "margin"].to_numpy(dtype=float)

    # Strictly train-only preprocessing.
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    abs_error = np.abs(y_test - y_pred)
    mae = float(mean_absolute_error(y_test, y_pred))

    test_df = work.loc[test_mask].copy()
    spread = pd.to_numeric(test_df.get("spread"), errors="coerce")
    keep = spread.notna()
    vegas_mae = np.nan
    vegas_n = 0
    model_mae_on_vegas_rows = np.nan
    if int(keep.sum()) > 0:
        y_true_market = pd.to_numeric(test_df.loc[keep, "margin"], errors="coerce").to_numpy(dtype=float)
        vegas_pred = -spread.loc[keep].to_numpy(dtype=float)
        vegas_mae = float(mean_absolute_error(y_true_market, vegas_pred))
        model_mae_on_vegas_rows = float(
            mean_absolute_error(y_true_market, y_pred[keep.to_numpy()])
        )
        vegas_n = int(keep.sum())

    return {
        "fold": fold.name,
        "train_n": train_n,
        "test_n": test_n,
        "alpha": float(alpha),
        "test_mae": mae,
        "test_abs_error": abs_error,
        "y_test": y_test,
        "y_pred": y_pred,
        "vegas_mae": vegas_mae,
        "vegas_n": vegas_n,
        "model_mae_on_vegas_rows": model_mae_on_vegas_rows,
    }


def main() -> None:
    base_df, torvik = build_base_model_frame()
    final_features = build_final_feature_list()
    select_folds = OUTER_FOLDS[:6]
    holdout_fold = OUTER_FOLDS[6]

    print(f"FINAL_BLOCKS: {FINAL_BLOCKS}")
    print(f"baseline: {BASELINE_FEATURES}")
    print(f"total_features_used: {len(final_features)}")

    fold_frames = {fold.name: preprocess_fold_df(base_df, torvik, fold) for fold in OUTER_FOLDS}

    # 1) Alpha sweep on folds 1-6.
    sweep_rows: List[Dict[str, object]] = []
    best_alpha = None
    best_pooled_mae = np.inf
    for alpha in ALPHA_GRID:
        per_fold_mae: Dict[str, float] = {}
        pooled_abs: List[np.ndarray] = []
        for fold in select_folds:
            res = fit_fixed_alpha_on_fold(
                fold_df=fold_frames[fold.name],
                fold=fold,
                feature_cols=final_features,
                alpha=float(alpha),
            )
            per_fold_mae[fold.name] = float(res["test_mae"])
            pooled_abs.append(np.asarray(res["test_abs_error"], dtype=float))

        pooled_mae = float(np.mean(np.concatenate(pooled_abs)))
        row = {"alpha": float(alpha), "pooled_mae_select": pooled_mae}
        for fold in select_folds:
            row[f"{fold.name.lower().replace(' ', '_')}_mae"] = per_fold_mae[fold.name]
        sweep_rows.append(row)

        if pooled_mae < best_pooled_mae:
            best_pooled_mae = pooled_mae
            best_alpha = float(alpha)

    sweep_df = pd.DataFrame(sweep_rows).sort_values("pooled_mae_select").reset_index(drop=True)
    sweep_df.to_csv(ALPHA_SWEEP_OUT, index=False)

    # 2) Validate best alpha on fold 7.
    holdout_result = fit_fixed_alpha_on_fold(
        fold_df=fold_frames[holdout_fold.name],
        fold=holdout_fold,
        feature_cols=final_features,
        alpha=float(best_alpha),
    )

    # 3) Final all-fold pooled MAE vs Vegas using chosen alpha.
    fold_rows: List[Dict[str, object]] = []
    pooled_model_abs: List[np.ndarray] = []
    pooled_model_vs_vegas_abs: List[np.ndarray] = []
    pooled_vegas_abs: List[np.ndarray] = []

    for fold in OUTER_FOLDS:
        res = fit_fixed_alpha_on_fold(
            fold_df=fold_frames[fold.name],
            fold=fold,
            feature_cols=final_features,
            alpha=float(best_alpha),
        )
        fold_rows.append(
            {
                "fold": res["fold"],
                "train_n": res["train_n"],
                "test_n": res["test_n"],
                "alpha": res["alpha"],
                "model_mae": res["test_mae"],
                "vegas_mae": res["vegas_mae"],
                "model_mae_on_vegas_rows": res["model_mae_on_vegas_rows"],
                "vegas_n": res["vegas_n"],
            }
        )
        pooled_model_abs.append(np.asarray(res["test_abs_error"], dtype=float))

        if np.isfinite(res["vegas_mae"]) and res["vegas_n"] > 0:
            # Reconstruct vegas-row absolute errors from MAE*count to avoid carrying full vectors around.
            vegas_abs_mean = float(res["vegas_mae"])
            model_vs_vegas_abs_mean = float(res["model_mae_on_vegas_rows"])
            pooled_vegas_abs.append(np.full(res["vegas_n"], vegas_abs_mean, dtype=float))
            pooled_model_vs_vegas_abs.append(np.full(res["vegas_n"], model_vs_vegas_abs_mean, dtype=float))

    fold_results_df = pd.DataFrame(fold_rows)
    fold_results_df.to_csv(FOLD_RESULTS_OUT, index=False)

    pooled_model_mae_all = float(np.mean(np.concatenate(pooled_model_abs)))
    pooled_vegas_mae_common = (
        float(np.mean(np.concatenate(pooled_vegas_abs))) if len(pooled_vegas_abs) > 0 else np.nan
    )
    pooled_model_mae_common = (
        float(np.mean(np.concatenate(pooled_model_vs_vegas_abs)))
        if len(pooled_model_vs_vegas_abs) > 0
        else np.nan
    )

    print("\nAlpha sweep (select folds 1-6)")
    print(f"best_alpha: {best_alpha:.6f}")
    print(f"best_pooled_mae_select: {best_pooled_mae:.6f}")
    print(f"holdout_fold7_mae: {holdout_result['test_mae']:.6f}")

    print("\nFinal model (chosen alpha) pooled over folds 1-7")
    print(f"pooled_model_mae_all_rows: {pooled_model_mae_all:.6f}")
    if np.isfinite(pooled_vegas_mae_common):
        print(f"pooled_model_mae_on_vegas_rows: {pooled_model_mae_common:.6f}")
        print(f"pooled_vegas_mae: {pooled_vegas_mae_common:.6f}")
        print(
            f"pooled_delta_model_minus_vegas: "
            f"{(pooled_model_mae_common - pooled_vegas_mae_common):+.6f}"
        )

    print(f"\nsaved: {ALPHA_SWEEP_OUT}")
    print(f"saved: {FOLD_RESULTS_OUT}")


if __name__ == "__main__":
    main()
