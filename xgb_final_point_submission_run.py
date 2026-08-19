"""Final point submission run using current ridge/XGB pipeline + locked blends."""

import argparse
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prediction_intervals import (
    PRE_XGB_REQUIRED,
    SOURCE_STATS,
    XGB_BASE_MARGIN_COL,
    XGB_POINT_FEATURES,
    add_ridge_dependent_xgb_features,
    fit_xgb_point_model,
    load_data,
    merge_torvik_asof,
    overwrite_rest_from_schedule,
)
from ridge_final_blocks_alpha_sweep import BASELINE_FEATURES as FINAL_BASELINE_FEATURES
from ridge_final_blocks_alpha_sweep import FINAL_BLOCKS, build_final_feature_list
from ridge_model import add_engineered_features, calculate_rest_days


MIYA_BLEND_W = 0.70
VEGAS_BLEND_W = 0.50
XGB_INTERCEPT_SHIFT = 1.0
AGE_COL_CANDIDATES = ("miya_age_days", "evan_age_days", "evan_spread_age_days")
DEFAULT_OUT_PATH = os.path.join("data", "xgb_future_acc_predictions_blend_w065.csv")
ROUND_PRED_DECIMALS = 3
FINAL_BLOCK_ALPHA_SWEEP_PATH = os.path.join("data", "final_blocks_alpha_sweep.csv")
RIDGE_EXTRA_STATS = ("efgd", "ftr", "ftrd")
RIDGE_FINAL_FEATURES = list(build_final_feature_list())
RIDGE_MERGE_STATS = list(dict.fromkeys(list(SOURCE_STATS) + list(RIDGE_EXTRA_STATS)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train current ridge/XGB models and produce final point predictions with "
            "locked blend: first xgb_intercept+miya, then 50/50 with Vegas when Vegas is available."
        )
    )
    parser.add_argument("--out-path", default=DEFAULT_OUT_PATH)
    parser.add_argument(
        "--future-games-path",
        default=None,
        help="Optional override path for future games CSV (can include evan_spread/miya columns).",
    )
    parser.add_argument(
        "--max-miya-age-days",
        type=float,
        default=None,
        help="Optional stale cutoff for Miya rows if a Miya age column exists.",
    )
    return parser.parse_args()


def choose_miya_source_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("evan_spread", "miya_spread", "miya_raw"):
        if c in df.columns:
            return c
    return None


def compute_miya_raw(df: pd.DataFrame, miya_col: Optional[str]) -> pd.Series:
    if miya_col is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    if miya_col == "evan_spread":
        # Spread orientation (home spread) to home-margin orientation.
        return -pd.to_numeric(df[miya_col], errors="coerce")
    return pd.to_numeric(df[miya_col], errors="coerce")


def build_miya_usable(
    df: pd.DataFrame, max_miya_age_days: Optional[float]
) -> Tuple[pd.Series, Optional[str], int]:
    usable = df["miya_raw"].notna().copy()
    if max_miya_age_days is None:
        return usable, None, 0

    for col in AGE_COL_CANDIDATES:
        if col not in df.columns:
            continue
        ages = pd.to_numeric(df[col], errors="coerce")
        stale = ages > float(max_miya_age_days)
        stale_n = int((usable & stale).sum())
        usable = usable & (~stale)
        return usable, col, stale_n
    return usable, None, 0


def load_data_with_optional_future_override(
    future_games_path: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_games, torvik, future_games = load_data()
    if future_games_path is None:
        return all_games, torvik, future_games

    future_games = pd.read_csv(future_games_path)
    future_games["home_team_id"] = pd.to_numeric(future_games["home_team_id"], errors="coerce")
    future_games["away_team_id"] = pd.to_numeric(future_games["away_team_id"], errors="coerce")
    future_games = future_games.dropna(subset=["home_team_id", "away_team_id"]).copy()
    future_games["home_team_id"] = future_games["home_team_id"].astype(int)
    future_games["away_team_id"] = future_games["away_team_id"].astype(int)
    future_games["date_dt"] = pd.to_datetime(future_games["date"], errors="coerce")
    future_games = future_games.dropna(subset=["date_dt"]).copy()

    mapped_ids = set(torvik["team_id"].astype(int))
    future_games = future_games[
        future_games["home_team_id"].isin(mapped_ids) & future_games["away_team_id"].isin(mapped_ids)
    ].copy()
    return all_games, torvik, future_games


def load_best_final_blocks_alpha(path: str = FINAL_BLOCK_ALPHA_SWEEP_PATH) -> float:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing alpha sweep file: {path}")
    sweep = pd.read_csv(path)
    if "alpha" not in sweep.columns or "pooled_mae_select" not in sweep.columns:
        raise ValueError(
            f"{path} must contain columns: alpha, pooled_mae_select"
        )
    work = sweep[["alpha", "pooled_mae_select"]].copy()
    work["alpha"] = pd.to_numeric(work["alpha"], errors="coerce")
    work["pooled_mae_select"] = pd.to_numeric(work["pooled_mae_select"], errors="coerce")
    work = work.dropna(subset=["alpha", "pooled_mae_select"]).copy()
    if len(work) == 0:
        raise ValueError(f"No valid alpha rows in: {path}")
    best_idx = work["pooled_mae_select"].idxmin()
    return float(work.loc[best_idx, "alpha"])


def ensure_feature_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    return out


def add_xgb_bridge_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add feature-name bridges required by PRE_XGB_REQUIRED/XGB_POINT_FEATURES."""
    out = df.copy()

    if "tempo_avg" not in out.columns and "adj_tempo_avg" in out.columns:
        out["tempo_avg"] = pd.to_numeric(out["adj_tempo_avg"], errors="coerce")

    if "abs_adj_o_diff" not in out.columns:
        out["abs_adj_o_diff"] = pd.to_numeric(out.get("adj_o_diff"), errors="coerce").abs()
    if "abs_adj_d_diff" not in out.columns:
        out["abs_adj_d_diff"] = pd.to_numeric(out.get("adj_d_diff"), errors="coerce").abs()
    if "abs_efg_diff" not in out.columns:
        out["abs_efg_diff"] = pd.to_numeric(out.get("efg_diff"), errors="coerce").abs()

    if "imbalance_diff" not in out.columns:
        home_adj_o = pd.to_numeric(out.get("home_adj_o"), errors="coerce")
        home_adj_d = pd.to_numeric(out.get("home_adj_d"), errors="coerce")
        away_adj_o = pd.to_numeric(out.get("away_adj_o"), errors="coerce")
        away_adj_d = pd.to_numeric(out.get("away_adj_d"), errors="coerce")
        out["od_imbalance_home"] = (home_adj_o - home_adj_d).abs()
        out["od_imbalance_away"] = (away_adj_o - away_adj_d).abs()
        out["imbalance_diff"] = out["od_imbalance_home"] - out["od_imbalance_away"]

    return out


def build_hist_frame_for_final_block_ridge(all_games: pd.DataFrame, torvik: pd.DataFrame) -> pd.DataFrame:
    hist = all_games.copy()
    hist["margin"] = pd.to_numeric(hist["margin"], errors="coerce")
    hist = hist.dropna(subset=["margin"]).copy()
    hist = calculate_rest_days(hist, all_games)
    merged = merge_torvik_asof(hist, torvik, RIDGE_MERGE_STATS)
    feat = add_engineered_features(merged)
    feat = add_xgb_bridge_features(feat)
    feat = ensure_feature_columns(feat, RIDGE_FINAL_FEATURES)
    feat = feat.sort_values("date_dt").reset_index(drop=True)
    return feat


def fit_final_block_ridge(hist_df: pd.DataFrame, alpha: float) -> Tuple[pd.DataFrame, Pipeline]:
    work = hist_df.copy()
    X = work[RIDGE_FINAL_FEATURES]
    y = pd.to_numeric(work["margin"], errors="coerce").to_numpy(dtype=float)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )
    model.fit(X, y)
    work["ridge_pred_spread"] = model.predict(X)
    return work, model


def build_future_frame_with_current_models(future_games_path: Optional[str]) -> Tuple[pd.DataFrame, float]:
    all_games, torvik, future_games = load_data_with_optional_future_override(future_games_path)
    if len(future_games) == 0:
        raise RuntimeError("No rows found in data/future_acc_games.csv")

    ridge_alpha = load_best_final_blocks_alpha()
    hist_df = build_hist_frame_for_final_block_ridge(all_games, torvik)
    hist_df = hist_df.dropna(subset=["margin"] + PRE_XGB_REQUIRED).copy()
    hist_df, ridge_model = fit_final_block_ridge(hist_df, ridge_alpha)
    hist_df = add_ridge_dependent_xgb_features(hist_df)
    hist_df = hist_df.dropna(subset=[XGB_BASE_MARGIN_COL] + XGB_POINT_FEATURES).copy()
    hist_df = hist_df.sort_values("date_dt").reset_index(drop=True)

    xgb_model = fit_xgb_point_model(
        hist_df[XGB_POINT_FEATURES].values,
        hist_df["margin"].values,
        pd.to_numeric(hist_df[XGB_BASE_MARGIN_COL], errors="coerce").values,
    )

    full_schedule = pd.concat(
        [
            all_games[["date", "home_team_id", "away_team_id"]],
            future_games[["date", "home_team_id", "away_team_id"]],
        ],
        ignore_index=True,
    )
    future_seed = calculate_rest_days(future_games.copy(), full_schedule)
    future_feat = merge_torvik_asof(future_seed, torvik, RIDGE_MERGE_STATS)
    future_schedule = pd.concat(
        [
            all_games[["date_dt", "home_team_id", "away_team_id"]],
            future_games[["date_dt", "home_team_id", "away_team_id"]],
        ],
        ignore_index=True,
    )
    future_feat = overwrite_rest_from_schedule(future_feat, future_schedule)
    future_feat = add_engineered_features(future_feat)
    future_feat = add_xgb_bridge_features(future_feat)
    future_feat = ensure_feature_columns(future_feat, RIDGE_FINAL_FEATURES)
    future_feat = future_feat.dropna(subset=PRE_XGB_REQUIRED).copy()
    future_feat = future_feat.sort_values("date_dt").reset_index(drop=True)

    future_feat["ridge_pred_spread"] = ridge_model.predict(future_feat[RIDGE_FINAL_FEATURES])
    future_feat = add_ridge_dependent_xgb_features(future_feat)
    future_feat = future_feat.dropna(subset=[XGB_BASE_MARGIN_COL] + XGB_POINT_FEATURES).copy()

    pred_raw_model = xgb_model.predict(
        future_feat[XGB_POINT_FEATURES].values,
        base_margin=pd.to_numeric(future_feat[XGB_BASE_MARGIN_COL], errors="coerce").values,
    )
    future_feat["pred_raw"] = pred_raw_model
    return future_feat, ridge_alpha


def main() -> None:
    args = parse_args()
    out, ridge_alpha = build_future_frame_with_current_models(args.future_games_path)

    miya_src_col = choose_miya_source_col(out)
    out["xgb_base_intercept"] = pd.to_numeric(out["pred_raw"], errors="coerce") - XGB_INTERCEPT_SHIFT
    out["miya_raw"] = compute_miya_raw(out, miya_src_col)
    out["miya_usable"], stale_col, stale_n = build_miya_usable(out, args.max_miya_age_days)

    # Stage 1: XGB intercept + Miya blend (fallback to XGB intercept when Miya unavailable).
    out["pred_xgb_miya_blend"] = out["xgb_base_intercept"]
    use_miya = out["miya_usable"].to_numpy(dtype=bool)
    out.loc[use_miya, "pred_xgb_miya_blend"] = (
        MIYA_BLEND_W * out.loc[use_miya, "xgb_base_intercept"] + (1.0 - MIYA_BLEND_W) * out.loc[use_miya, "miya_raw"]
    )

    # Stage 2: If Vegas spread exists, blend 50/50 with stage-1 prediction.
    if "spread" in out.columns:
        out["vegas_pred"] = -pd.to_numeric(out["spread"], errors="coerce")
    else:
        out["vegas_pred"] = np.nan
    out["vegas_usable"] = out["vegas_pred"].notna()

    out["pred_final_submission"] = out["pred_xgb_miya_blend"]
    use_vegas = out["vegas_usable"].to_numpy(dtype=bool)
    out.loc[use_vegas, "pred_final_submission"] = (
        VEGAS_BLEND_W * out.loc[use_vegas, "pred_xgb_miya_blend"]
        + (1.0 - VEGAS_BLEND_W) * out.loc[use_vegas, "vegas_pred"]
    )
    # Backward-compat alias for any downstream readers.
    out["pred_final_locked"] = out["pred_final_submission"]

    cols = [
        "event_id",
        "date",
        "home_team_id",
        "away_team_id",
        "pred_raw",
        "xgb_base_intercept",
        "miya_raw",
        "miya_usable",
        "pred_xgb_miya_blend",
        "vegas_pred",
        "vegas_usable",
        "pred_final_submission",
        "pred_final_locked",
        "ridge_pred_spread",
    ]
    if "spread" in out.columns:
        cols.append("spread")

    round_cols = [
        "pred_raw",
        "xgb_base_intercept",
        "miya_raw",
        "pred_xgb_miya_blend",
        "vegas_pred",
        "pred_final_submission",
        "pred_final_locked",
        "ridge_pred_spread",
    ]
    for col in round_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(ROUND_PRED_DECIMALS)

    out[cols].to_csv(args.out_path, index=False)

    print("Final point submission blend complete.")
    print("model_source: prediction_intervals.py + ridge_final_blocks_alpha_sweep")
    print(f"ridge_final_blocks: {FINAL_BLOCKS}")
    print(f"ridge_baseline_features: {FINAL_BASELINE_FEATURES}")
    print(f"ridge_n_features: {len(RIDGE_FINAL_FEATURES)}")
    print(f"ridge_best_alpha_from_sweep: {ridge_alpha:.6f}")
    print(f"ridge_alpha_source: {FINAL_BLOCK_ALPHA_SWEEP_PATH}")
    print(f"miya_blend_w: {MIYA_BLEND_W:.2f}")
    print(f"vegas_blend_w: {VEGAS_BLEND_W:.2f}")
    print(f"xgb_intercept_shift: {XGB_INTERCEPT_SHIFT:.2f}")
    print(f"miya_source_col: {miya_src_col if miya_src_col is not None else 'none_found'}")
    print(f"rows_total: {len(out)}")
    print(f"miya_usable_rows: {int(out['miya_usable'].sum())}")
    print(f"vegas_usable_rows: {int(out['vegas_usable'].sum())}")
    print(f"rounded_prediction_decimals: {ROUND_PRED_DECIMALS}")
    if stale_col is not None:
        print(f"staleness_col_used: {stale_col}")
        print(f"miya_stale_rows_excluded: {stale_n}")
    elif args.max_miya_age_days is not None:
        print("warning: max-miya-age-days set but no Miya age column was found; stale filtering not applied.")
    print(f"saved: {args.out_path}")


if __name__ == "__main__":
    main()
