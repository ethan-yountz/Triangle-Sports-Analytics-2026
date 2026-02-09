"""Build fixed-cutoff spread prediction intervals and optional future-game forecasts."""

import argparse
import heapq
import os
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


TORVIK_PATH = os.path.join("data", "torvik_asof_ratings_all_teams.csv")
ALL_GAMES_PATH = os.path.join("data", "all_games.csv")
FUTURE_GAMES_PATH = os.path.join("data", "future_acc_games.csv")
OUTPUT_PATH = os.path.join("data", "predictions_with_intervals.csv")

SOURCE_STATS = [
    "adj_o",
    "adj_d",
    "efg",
    "tor",
    "tord",
    "orb",
    "drb",
    "three_pt_rate",
    "three_pt_pct",
    "adj_tempo",
    "days_since_last_game",
    "games_last_7_days",
]

RIDGE_FEATURES = [
    "adj_o_diff",
    "adj_d_diff",
    "efg_diff",
    "is_neutral",
    "tor_diff",
    "tord_diff",
    "orb_diff",
    "drb_diff",
    "days_since_last_game_diff",
    "games_last_7_days_diff",
    "fatigue_index_diff",
]

XGB_POINT_FEATURES = [
    "ridge_pred_spread",
    "abs_adj_o_diff",
    "abs_adj_d_diff",
    "abs_efg_diff",
    "imbalance_diff",
]

PRE_XGB_REQUIRED = [
    "abs_adj_o_diff",
    "abs_adj_d_diff",
    "abs_efg_diff",
    "imbalance_diff",
    "tempo_avg",
    "three_pt_rate_avg",
]

SPREAD_BUCKETS = [
    (0.0, 5.0),
    (5.0, 12.0),
    (12.0, 20.0),
    (20.0, 34.0),
    (34.0, np.inf),
]
SPREAD_BUCKET_NAMES = ["B0_<5", "B1_5_12", "B2_12_20", "B3_20_34", "B4_34p"]
NUM_SPREAD_BUCKETS = len(SPREAD_BUCKETS)

WIDTH_CAP = 30.0
MAX_CDF_T = 40
GREEDY_STEP = 0.5
MIN_FULL_BUCKET_N = 35
MIN_ST_BUCKET_N = 25
MIN_S_BUCKET_N = 20
MIN_IMB_GROUP_N = 120
MIN_DELTA = 1e-4
BLOWOUT_DAMP_C = 11.0
BLOWOUT_DAMP_ALPHA = 0.45
ROUNDING_AWARE_CALIBRATION = True
CROSS_TIME_CALIB_FOLDS = 3
MIN_CALIB_FOLD_SIZE = 80
USE_MULTI_CUTOFF_STABILITY = True
MULTI_CUTOFF_FRACTIONS = [(0.40, 0.56), (0.46, 0.62), (0.52, 0.68)]
MULTI_CUTOFF_MULT_GRID = np.arange(0.90, 1.101, 0.005)


@dataclass
class BucketProfile:
    n: int
    median_abs: float
    q87_abs: float
    cdf: np.ndarray


@dataclass
class WidthAllocator:
    use_imbalance: bool
    tempo_median: float
    imbalance_median: float
    tempo_cut: float
    uncertainty_scaler: Optional[StandardScaler]
    uncertainty_model: Optional[Ridge]
    base_c: float
    calibration_mult: float
    expected_target: float
    target_coverage: float
    width_cap: float
    max_cdf_t: int
    spread_post_multipliers: np.ndarray
    pre_blowout_mult: float
    tempo_low_mult: float
    tempo_high_mult: float
    blowout_cap: Optional[float]
    full_profiles: Dict[Tuple[int, int, int], BucketProfile]
    st_profiles: Dict[Tuple[int, int], BucketProfile]
    s_profiles: Dict[int, BucketProfile]
    global_profile: BucketProfile
    calibration_summary: Dict[str, float]


def quantile_higher(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="higher"))


def round_interval_bounds(lower: np.ndarray, upper: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Submission-style integer bounds: ceil(lower), floor(upper)."""
    lb = np.ceil(np.asarray(lower, dtype=float))
    ub = np.floor(np.asarray(upper, dtype=float))
    bad = ub < lb
    if np.any(bad):
        ub[bad] = lb[bad]
    return lb, ub


def coverage_aiw_from_mu_width(
    mu: np.ndarray,
    y_true: np.ndarray,
    widths: np.ndarray,
    rounded: bool = False,
) -> Tuple[float, float]:
    mu = np.asarray(mu, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    widths = np.asarray(widths, dtype=float)
    lower = mu - widths
    upper = mu + widths
    if rounded:
        lower, upper = round_interval_bounds(lower, upper)
    covered = (y_true >= lower) & (y_true <= upper)
    coverage = float(np.mean(covered))
    aiw = float(np.mean(upper - lower))
    return coverage, aiw


def build_time_folds(
    n_rows: int,
    n_folds: int = CROSS_TIME_CALIB_FOLDS,
    min_fold_size: int = MIN_CALIB_FOLD_SIZE,
) -> List[np.ndarray]:
    if n_rows <= 0:
        return []
    max_folds = max(1, min(n_folds, n_rows // max(1, min_fold_size)))
    if max_folds <= 1:
        return [np.arange(n_rows)]
    raw = [np.asarray(x, dtype=int) for x in np.array_split(np.arange(n_rows), max_folds)]
    folds: List[np.ndarray] = []
    for part in raw:
        if len(part) == 0:
            continue
        if len(part) < min_fold_size and len(folds) > 0:
            folds[-1] = np.concatenate([folds[-1], part])
        else:
            folds.append(part)
    if len(folds) > 1 and len(folds[-1]) < min_fold_size:
        folds[-2] = np.concatenate([folds[-2], folds[-1]])
        folds = folds[:-1]
    return folds if folds else [np.arange(n_rows)]


def fold_calibration_score(
    mu: np.ndarray,
    y_true: np.ndarray,
    widths: np.ndarray,
    target_coverage: float,
    folds: List[np.ndarray],
    rounded: bool = ROUNDING_AWARE_CALIBRATION,
) -> Tuple[float, float, List[float], float]:
    cov_all, aiw_all = coverage_aiw_from_mu_width(mu, y_true, widths, rounded=rounded)
    fold_covs: List[float] = []
    for idx in folds:
        if len(idx) == 0:
            continue
        cov_f, _ = coverage_aiw_from_mu_width(mu[idx], y_true[idx], widths[idx], rounded=rounded)
        fold_covs.append(cov_f)
    if len(fold_covs) == 0:
        fold_covs = [cov_all]
    under_pen = float(np.mean([max(0.0, target_coverage - c) for c in fold_covs]))
    gap = abs(cov_all - target_coverage) + (0.002 if cov_all < target_coverage else 0.0)
    score = gap + 0.5 * under_pen
    return cov_all, aiw_all, fold_covs, score


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_games = pd.read_csv(ALL_GAMES_PATH)
    torvik = pd.read_csv(TORVIK_PATH)
    future_games = pd.read_csv(FUTURE_GAMES_PATH) if os.path.exists(FUTURE_GAMES_PATH) else pd.DataFrame()

    torvik["team_id"] = pd.to_numeric(torvik["team_id"], errors="coerce")
    torvik = torvik[torvik["team_id"].notna()].copy()
    torvik["team_id"] = torvik["team_id"].astype(int)
    torvik["date_dt"] = pd.to_datetime(torvik["date"], errors="coerce")
    torvik = torvik.dropna(subset=["date_dt"]).copy()

    for df in [all_games, future_games]:
        if len(df) == 0:
            continue
        df["home_team_id"] = pd.to_numeric(df["home_team_id"], errors="coerce")
        df["away_team_id"] = pd.to_numeric(df["away_team_id"], errors="coerce")
        df.dropna(subset=["home_team_id", "away_team_id"], inplace=True)
        df["home_team_id"] = df["home_team_id"].astype(int)
        df["away_team_id"] = df["away_team_id"].astype(int)
        df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date_dt"], inplace=True)

    mapped_ids = set(torvik["team_id"].astype(int))
    all_games = all_games[
        all_games["home_team_id"].isin(mapped_ids) & all_games["away_team_id"].isin(mapped_ids)
    ].copy()
    future_games = future_games[
        future_games["home_team_id"].isin(mapped_ids) & future_games["away_team_id"].isin(mapped_ids)
    ].copy()
    return all_games, torvik, future_games


def _merge_side_asof(
    games: pd.DataFrame,
    torvik: pd.DataFrame,
    team_col: str,
    prefix: str,
    stat_cols: List[str],
) -> pd.DataFrame:
    left = games[["row_id", team_col, "as_of_date_dt"]].copy()
    right = torvik[["team_id", "date_dt"] + stat_cols].copy()
    right = right.rename(columns={"team_id": team_col, "date_dt": "torvik_date_dt"})

    parts = []
    for team_id, left_grp in left.groupby(team_col, sort=False):
        left_grp = left_grp.sort_values("as_of_date_dt")
        right_grp = right[right[team_col] == team_id].sort_values("torvik_date_dt")
        if len(right_grp) == 0:
            empty = left_grp.copy()
            for c in stat_cols:
                empty[c] = np.nan
            parts.append(empty)
            continue

        merged_grp = pd.merge_asof(
            left_grp,
            right_grp,
            left_on="as_of_date_dt",
            right_on="torvik_date_dt",
            direction="backward",
        )
        parts.append(merged_grp)

    merged = pd.concat(parts, ignore_index=True)
    rename_map = {c: f"{prefix}{c}" for c in stat_cols}
    merged = merged.rename(columns=rename_map)
    keep_cols = ["row_id"] + list(rename_map.values())
    return merged[keep_cols]


def merge_torvik_asof(games: pd.DataFrame, torvik: pd.DataFrame, stat_cols: List[str]) -> pd.DataFrame:
    out = games.copy().reset_index(drop=True)
    out["row_id"] = np.arange(len(out))
    out["as_of_date_dt"] = out["date_dt"] - pd.Timedelta(days=1)

    home = _merge_side_asof(out, torvik, "home_team_id", "home_", stat_cols)
    away = _merge_side_asof(out, torvik, "away_team_id", "away_", stat_cols)

    out = out.merge(home, on="row_id", how="left")
    out = out.merge(away, on="row_id", how="left")
    return out.drop(columns=["row_id"])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["adj_o_diff"] = out["home_adj_o"] - out["away_adj_o"]
    out["adj_d_diff"] = out["home_adj_d"] - out["away_adj_d"]
    out["efg_diff"] = out["home_efg"] - out["away_efg"]
    out["is_neutral"] = out["neutral_site"].astype(int)

    out["tor_diff"] = out["home_tor"] - out["away_tor"]
    out["tord_diff"] = out["home_tord"] - out["away_tord"]
    out["orb_diff"] = out["home_orb"] - out["away_orb"]
    out["drb_diff"] = out["home_drb"] - out["away_drb"]

    out["home_days_since_last_game"] = pd.to_numeric(out["home_days_since_last_game"], errors="coerce")
    out["away_days_since_last_game"] = pd.to_numeric(out["away_days_since_last_game"], errors="coerce")
    out["home_games_last_7_days"] = pd.to_numeric(out["home_games_last_7_days"], errors="coerce")
    out["away_games_last_7_days"] = pd.to_numeric(out["away_games_last_7_days"], errors="coerce")

    out["days_since_last_game_diff"] = out["home_days_since_last_game"] - out["away_days_since_last_game"]
    out["games_last_7_days_diff"] = out["home_games_last_7_days"] - out["away_games_last_7_days"]
    out["home_fatigue_index"] = out["home_games_last_7_days"] / np.maximum(out["home_days_since_last_game"], 1.0)
    out["away_fatigue_index"] = out["away_games_last_7_days"] / np.maximum(out["away_days_since_last_game"], 1.0)
    out["fatigue_index_diff"] = out["home_fatigue_index"] - out["away_fatigue_index"]

    out["abs_adj_o_diff"] = out["adj_o_diff"].abs()
    out["abs_adj_d_diff"] = out["adj_d_diff"].abs()
    out["abs_efg_diff"] = out["efg_diff"].abs()
    out["od_imbalance_home"] = (out["home_adj_o"] - out["home_adj_d"]).abs()
    out["od_imbalance_away"] = (out["away_adj_o"] - out["away_adj_d"]).abs()
    out["imbalance_diff"] = out["od_imbalance_home"] - out["od_imbalance_away"]

    out["tempo_avg"] = (out["home_adj_tempo"] + out["away_adj_tempo"]) / 2.0
    out["three_pt_rate_avg"] = (out["home_three_pt_rate"] + out["away_three_pt_rate"]) / 2.0
    return out


def overwrite_rest_from_schedule(
    target_games: pd.DataFrame,
    schedule_games: pd.DataFrame,
) -> pd.DataFrame:
    """Overwrite rest/congestion fields using known schedule dates."""
    out = target_games.copy()

    sched = schedule_games[["date_dt", "home_team_id", "away_team_id"]].copy()
    sched["date_dt"] = pd.to_datetime(sched["date_dt"], errors="coerce")
    sched = sched.dropna(subset=["date_dt"]).copy()

    appearances = pd.concat(
        [
            sched[["home_team_id", "date_dt"]].rename(columns={"home_team_id": "team_id"}),
            sched[["away_team_id", "date_dt"]].rename(columns={"away_team_id": "team_id"}),
        ],
        ignore_index=True,
    )
    appearances["team_id"] = pd.to_numeric(appearances["team_id"], errors="coerce")
    appearances = appearances.dropna(subset=["team_id"]).copy()
    appearances["team_id"] = appearances["team_id"].astype(int)

    team_dates: Dict[int, np.ndarray] = {}
    for team_id, grp in appearances.groupby("team_id", sort=False):
        team_dates[int(team_id)] = np.sort(grp["date_dt"].to_numpy(dtype="datetime64[ns]"))

    home_days: List[float] = []
    away_days: List[float] = []
    home_games7: List[float] = []
    away_games7: List[float] = []

    for _, row in out.iterrows():
        game_date = pd.to_datetime(row["date_dt"]).to_datetime64()
        for side in ("home", "away"):
            team_id = int(row[f"{side}_team_id"])
            dates = team_dates.get(team_id, np.array([], dtype="datetime64[ns]"))
            idx = int(np.searchsorted(dates, game_date, side="left"))

            if idx == 0:
                days = 1.0
            else:
                days = float((game_date - dates[idx - 1]) / np.timedelta64(1, "D"))

            window_start = game_date - np.timedelta64(7, "D")
            lo = int(np.searchsorted(dates, window_start, side="left"))
            games7 = float(idx - lo)
            if games7 <= 0:
                games7 = 1.0

            if side == "home":
                home_days.append(days)
                home_games7.append(games7)
            else:
                away_days.append(days)
                away_games7.append(games7)

    out["home_days_since_last_game"] = home_days
    out["away_days_since_last_game"] = away_days
    out["home_games_last_7_days"] = home_games7
    out["away_games_last_7_days"] = away_games7
    return out


def apply_fixed_cutoff_snapshot(
    df: pd.DataFrame,
    torvik: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    stat_cols: List[str],
) -> pd.DataFrame:
    out = df.copy()
    future_mask = out["date_dt"] > cutoff_date
    if not future_mask.any():
        return out

    torvik_cut = torvik[torvik["date_dt"] <= cutoff_date].copy()
    torvik_cut = torvik_cut.sort_values("date_dt").groupby("team_id", as_index=False).tail(1)

    for stat in stat_cols:
        snapshot = torvik_cut.set_index("team_id")[stat].to_dict()
        out.loc[future_mask, f"home_{stat}"] = out.loc[future_mask, "home_team_id"].map(snapshot)
        out.loc[future_mask, f"away_{stat}"] = out.loc[future_mask, "away_team_id"].map(snapshot)
    return out


def choose_tscv_splits(n_rows: int) -> int:
    if n_rows >= 1000:
        return 5
    if n_rows >= 400:
        return 4
    if n_rows >= 200:
        return 3
    return 2


def fit_ridge_baseline(df: pd.DataFrame, train_mask: pd.Series) -> Tuple[pd.DataFrame, Ridge, StandardScaler]:
    out = df.copy()
    X_train = out.loc[train_mask, RIDGE_FEATURES].values
    y_train = out.loc[train_mask, "margin"].values
    X_all = out[RIDGE_FEATURES].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_all_s = scaler.transform(X_all)

    ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=TimeSeriesSplit(n_splits=choose_tscv_splits(len(X_train))))
    ridge_cv.fit(X_train_s, y_train)
    ridge = Ridge(alpha=float(ridge_cv.alpha_))
    ridge.fit(X_train_s, y_train)
    out["ridge_pred_spread"] = ridge.predict(X_all_s)
    return out, ridge, scaler


def fit_xgb_point_model(X_train: np.ndarray, y_train: np.ndarray) -> XGBRegressor:
    n_train = len(X_train)
    val_size = max(100, int(0.2 * n_train))
    if val_size >= n_train:
        val_size = max(20, n_train // 5)
    fit_size = n_train - val_size

    model = XGBRegressor(
        objective="reg:squarederror",
        max_depth=3,
        min_child_weight=8,
        learning_rate=0.05,
        n_estimators=400,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        eval_metric="mae",
        early_stopping_rounds=30,
    )

    X_fit = X_train[:fit_size]
    y_fit = y_train[:fit_size]
    X_val = X_train[fit_size:]
    y_val = y_train[fit_size:]
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    return model


def dampen_point_predictions(
    preds: np.ndarray,
    c: float = BLOWOUT_DAMP_C,
    alpha: float = BLOWOUT_DAMP_ALPHA,
) -> np.ndarray:
    preds = np.asarray(preds, dtype=float)
    abs_pred = np.abs(preds)
    damp_abs = np.minimum(abs_pred, c + alpha * abs_pred)
    return np.sign(preds) * damp_abs


def spread_bucket_index(abs_mu: np.ndarray) -> np.ndarray:
    out = np.zeros(len(abs_mu), dtype=int)
    for i, (lo, hi) in enumerate(SPREAD_BUCKETS):
        mask = abs_mu >= lo
        if np.isfinite(hi):
            mask &= abs_mu < hi
        out[mask] = i
    return out


def audit_spread_bucket_index(abs_mu: np.ndarray) -> np.ndarray:
    out = np.zeros(len(abs_mu), dtype=int)
    out[(abs_mu >= 5.0) & (abs_mu < 12.0)] = 1
    out[(abs_mu >= 12.0) & (abs_mu < 20.0)] = 2
    out[abs_mu >= 20.0] = 3
    return out


def build_profile(abs_residuals: np.ndarray, max_cdf_t: int) -> BucketProfile:
    vals = np.asarray(abs_residuals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        vals = np.array([10.0], dtype=float)
    median_abs = float(np.median(vals))
    q87_abs = quantile_higher(vals, 0.87)
    cdf = np.zeros(max_cdf_t + 1, dtype=float)
    for t in range(max_cdf_t + 1):
        cdf[t] = float(np.mean(vals <= t))
    return BucketProfile(n=int(len(vals)), median_abs=median_abs, q87_abs=q87_abs, cdf=cdf)


def cdf_lookup(profile: BucketProfile, width: float, max_cdf_t: int) -> float:
    if width <= 0:
        return float(profile.cdf[0])
    if width >= max_cdf_t:
        return float(profile.cdf[max_cdf_t])
    lo = int(np.floor(width))
    hi = int(np.ceil(width))
    lo = min(max(lo, 0), max_cdf_t)
    hi = min(max(hi, 0), max_cdf_t)
    if lo == hi:
        return float(profile.cdf[lo])
    frac = width - lo
    return float(profile.cdf[lo] + frac * (profile.cdf[hi] - profile.cdf[lo]))


def delta_one_point(
    profile: BucketProfile,
    width: float,
    width_cap: float,
    max_cdf_t: int,
    step: float,
) -> float:
    if width + step > width_cap + 1e-9:
        return 0.0
    p0 = cdf_lookup(profile, width, max_cdf_t)
    p1 = cdf_lookup(profile, width + step, max_cdf_t)
    return max(0.0, p1 - p0)


def row_bucket_codes(
    df: pd.DataFrame,
    mu: np.ndarray,
    tempo_median: float,
    imbalance_median: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    abs_mu = np.abs(np.asarray(mu, dtype=float))
    spread_code = spread_bucket_index(abs_mu)

    tempo = pd.to_numeric(df["tempo_avg"], errors="coerce").values
    tempo_fill = float(np.nanmedian(tempo[np.isfinite(tempo)])) if np.isfinite(tempo).any() else tempo_median
    tempo = np.where(np.isfinite(tempo), tempo, tempo_fill)
    tempo_code = (tempo > tempo_median).astype(int)

    imb = np.abs(pd.to_numeric(df["imbalance_diff"], errors="coerce").values)
    imb_fill = float(np.nanmedian(imb[np.isfinite(imb)])) if np.isfinite(imb).any() else imbalance_median
    imb = np.where(np.isfinite(imb), imb, imb_fill)
    imb_code = (imb > imbalance_median).astype(int)
    return spread_code, tempo_code, imb_code


def create_profiles_by_level(
    spread_code: np.ndarray,
    tempo_code: np.ndarray,
    imb_code: np.ndarray,
    abs_res: np.ndarray,
    use_imbalance: bool,
    max_cdf_t: int,
) -> Tuple[
    Dict[Tuple[int, int, int], BucketProfile],
    Dict[Tuple[int, int], BucketProfile],
    Dict[int, BucketProfile],
    BucketProfile,
]:
    full_profiles: Dict[Tuple[int, int, int], BucketProfile] = {}
    st_profiles: Dict[Tuple[int, int], BucketProfile] = {}
    s_profiles: Dict[int, BucketProfile] = {}
    global_profile = build_profile(abs_res, max_cdf_t)

    if use_imbalance:
        for s in range(NUM_SPREAD_BUCKETS):
            for t in range(2):
                for i in range(2):
                    mask = (spread_code == s) & (tempo_code == t) & (imb_code == i)
                    if int(mask.sum()) >= MIN_FULL_BUCKET_N:
                        full_profiles[(s, t, i)] = build_profile(abs_res[mask], max_cdf_t)

    for s in range(NUM_SPREAD_BUCKETS):
        for t in range(2):
            mask = (spread_code == s) & (tempo_code == t)
            if int(mask.sum()) >= MIN_ST_BUCKET_N:
                st_profiles[(s, t)] = build_profile(abs_res[mask], max_cdf_t)

    for s in range(NUM_SPREAD_BUCKETS):
        mask = spread_code == s
        if int(mask.sum()) >= MIN_S_BUCKET_N:
            s_profiles[s] = build_profile(abs_res[mask], max_cdf_t)

    return full_profiles, st_profiles, s_profiles, global_profile


def resolve_profile(
    allocator: WidthAllocator,
    spread_code: int,
    tempo_code: int,
    imb_code: int,
) -> BucketProfile:
    if allocator.use_imbalance:
        prof = allocator.full_profiles.get((spread_code, tempo_code, imb_code))
        if prof is not None:
            return prof

    prof_st = allocator.st_profiles.get((spread_code, tempo_code))
    if prof_st is not None:
        return prof_st

    prof_s = allocator.s_profiles.get(spread_code)
    if prof_s is not None:
        return prof_s

    return allocator.global_profile


def map_profiles_for_rows(
    df: pd.DataFrame,
    mu: np.ndarray,
    allocator: WidthAllocator,
) -> List[BucketProfile]:
    s_code, t_code, i_code = row_bucket_codes(
        df=df,
        mu=mu,
        tempo_median=allocator.tempo_median,
        imbalance_median=allocator.imbalance_median,
    )
    profiles: List[BucketProfile] = []
    for j in range(len(df)):
        profiles.append(resolve_profile(allocator, int(s_code[j]), int(t_code[j]), int(i_code[j])))
    return profiles


def expected_coverage_from_widths(
    widths: np.ndarray,
    profiles: List[BucketProfile],
    max_cdf_t: int,
) -> Tuple[np.ndarray, float]:
    p = np.array([cdf_lookup(profiles[i], float(widths[i]), max_cdf_t) for i in range(len(widths))], dtype=float)
    return p, float(p.mean())


def greedy_allocate_width(
    base_widths: np.ndarray,
    profiles: List[BucketProfile],
    expected_target: float,
    width_cap: float,
    max_cdf_t: int,
    step: float = GREEDY_STEP,
    min_delta: float = MIN_DELTA,
) -> Tuple[np.ndarray, float, int]:
    widths = np.asarray(base_widths, dtype=float).copy()
    widths = np.clip(widths, 0.5, width_cap)
    n = len(widths)

    probs, exp_cov = expected_coverage_from_widths(widths, profiles, max_cdf_t)
    if exp_cov >= expected_target:
        return widths, exp_cov, 0

    versions = np.zeros(n, dtype=int)
    heap: List[Tuple[float, int, int]] = []
    for i in range(n):
        d = delta_one_point(profiles[i], widths[i], width_cap, max_cdf_t, step)
        heapq.heappush(heap, (-d, i, versions[i]))

    added_points = 0
    max_steps = int(np.ceil(n * width_cap))

    while exp_cov < expected_target and heap and added_points < max_steps:
        neg_d, i, v = heapq.heappop(heap)
        if v != versions[i]:
            continue
        d = -neg_d
        if d <= min_delta:
            break
        if widths[i] + step > width_cap + 1e-9:
            continue

        old_p = probs[i]
        widths[i] += step
        new_p = cdf_lookup(profiles[i], widths[i], max_cdf_t)
        probs[i] = new_p
        exp_cov += (new_p - old_p) / n
        added_points += 1

        versions[i] += 1
        d_new = delta_one_point(profiles[i], widths[i], width_cap, max_cdf_t, step)
        heapq.heappush(heap, (-d_new, i, versions[i]))

    return widths, exp_cov, added_points


def choose_base_c(
    abs_err: np.ndarray,
    s_bucket: np.ndarray,
    target_coverage: float,
    width_cap: float,
) -> Tuple[float, float, float]:
    c_grid = np.arange(0.70, 2.81, 0.05)
    baseline_low = max(0.60, target_coverage - 0.03)
    baseline_high = max(baseline_low, target_coverage - 0.01)
    baseline_mid = (baseline_low + baseline_high) / 2.0

    rows = []
    for c in c_grid:
        w = np.clip(c * s_bucket, 0.5, width_cap)
        cov = float(np.mean(abs_err <= w))
        aiw = float(np.mean(2.0 * w))
        rows.append((c, cov, aiw))

    in_band = [r for r in rows if baseline_low <= r[1] <= baseline_high]
    if in_band:
        best = min(in_band, key=lambda x: x[2])
        return best[0], best[1], best[2]

    under = [r for r in rows if r[1] < baseline_low]
    if under:
        best = max(under, key=lambda x: x[1])
        return best[0], best[1], best[2]

    best = min(rows, key=lambda x: abs(x[1] - baseline_mid))
    return best[0], best[1], best[2]


def build_uncertainty_matrix(df: pd.DataFrame, mu: np.ndarray, tempo_cut: float) -> np.ndarray:
    abs_pred = np.abs(np.asarray(mu, dtype=float))
    tempo = pd.to_numeric(df["tempo_avg"], errors="coerce").values
    imbalance_abs = np.abs(pd.to_numeric(df["imbalance_diff"], errors="coerce").values)
    three_pt_rate_avg = pd.to_numeric(df["three_pt_rate_avg"], errors="coerce").values
    var_proxy = three_pt_rate_avg * tempo

    X = np.column_stack(
        [
            abs_pred,
            tempo,
            (tempo <= tempo_cut).astype(float),
            imbalance_abs,
            var_proxy,
        ]
    )
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = np.isfinite(col)
        fill = float(np.nanmedian(col[finite])) if finite.any() else 0.0
        col[~finite] = fill
        X[:, j] = col
    return X


def choose_base_multiplier(
    mu: np.ndarray,
    y_true: np.ndarray,
    base_hw: np.ndarray,
    target_coverage: float,
    width_cap: float,
    folds: List[np.ndarray],
    rounded: bool = ROUNDING_AWARE_CALIBRATION,
) -> Tuple[float, float, float]:
    m_grid = np.arange(0.70, 1.81, 0.02)
    baseline_low = max(0.60, target_coverage - 0.03)
    baseline_high = max(baseline_low, target_coverage - 0.01)
    baseline_mid = (baseline_low + baseline_high) / 2.0

    rows = []
    for m in m_grid:
        w = np.clip(base_hw * m, 0.5, width_cap)
        cov, aiw, _, score = fold_calibration_score(
            mu=mu,
            y_true=y_true,
            widths=w,
            target_coverage=target_coverage,
            folds=folds,
            rounded=rounded,
        )
        rows.append((m, cov, aiw, score))

    in_band = [r for r in rows if baseline_low <= r[1] <= baseline_high]
    if in_band:
        best = min(in_band, key=lambda x: (x[3], x[2]))
        return best[0], best[1], best[2]

    under = [r for r in rows if r[1] < baseline_low]
    if under:
        best = max(under, key=lambda x: x[1])
        return best[0], best[1], best[2]

    best = min(rows, key=lambda x: (abs(x[1] - baseline_mid), x[3], x[2]))
    return best[0], best[1], best[2]


def aggregate_allocators(allocators: List[WidthAllocator], template: WidthAllocator) -> WidthAllocator:
    if len(allocators) == 0:
        return template

    spread_stack = np.vstack([a.spread_post_multipliers for a in allocators])
    cap_vals = [a.blowout_cap for a in allocators if a.blowout_cap is not None]
    cap = float(np.median(cap_vals)) if len(cap_vals) > 0 else None

    return replace(
        template,
        base_c=float(np.median([a.base_c for a in allocators])),
        calibration_mult=float(np.median([a.calibration_mult for a in allocators])),
        expected_target=float(np.median([a.expected_target for a in allocators])),
        pre_blowout_mult=float(np.median([a.pre_blowout_mult for a in allocators])),
        tempo_low_mult=float(np.median([a.tempo_low_mult for a in allocators])),
        tempo_high_mult=float(np.median([a.tempo_high_mult for a in allocators])),
        spread_post_multipliers=np.asarray(np.median(spread_stack, axis=0), dtype=float),
        blowout_cap=cap,
    )


def tune_allocator_scalar_on_calib(
    allocator: WidthAllocator,
    calib_df: pd.DataFrame,
    mu_calib: np.ndarray,
    y_calib: np.ndarray,
    target_coverage: float,
) -> Tuple[WidthAllocator, Dict[str, float]]:
    best_ok: Optional[Tuple[float, float, float, float]] = None  # (aiw, gap, m, cov)
    best_any: Optional[Tuple[float, float, float]] = None  # (gap, m, cov)

    for m in MULTI_CUTOFF_MULT_GRID:
        cand = replace(allocator, calibration_mult=float(allocator.calibration_mult * float(m)))
        hw, _ = predict_half_widths(calib_df, mu_calib, cand)
        cov, aiw = evaluate_intervals(
            y_calib,
            mu_calib - hw,
            mu_calib + hw,
            rounded=ROUNDING_AWARE_CALIBRATION,
        )
        gap = abs(cov - target_coverage)
        if best_any is None or gap < best_any[0] - 1e-12:
            best_any = (gap, float(m), cov)
        if cov >= target_coverage:
            rec = (aiw, gap, float(m), cov)
            if best_ok is None or rec < best_ok:
                best_ok = rec

    if best_ok is not None:
        chosen_m = best_ok[2]
    elif best_any is not None:
        chosen_m = best_any[1]
    else:
        chosen_m = 1.0

    tuned = replace(allocator, calibration_mult=float(allocator.calibration_mult * chosen_m))
    hw_tuned, _ = predict_half_widths(calib_df, mu_calib, tuned)
    cov_tuned, aiw_tuned = evaluate_intervals(
        y_calib,
        mu_calib - hw_tuned,
        mu_calib + hw_tuned,
        rounded=ROUNDING_AWARE_CALIBRATION,
    )
    summary = {
        "multi_cutoff_scalar_m": float(chosen_m),
        "multi_cutoff_scalar_cov_calib": float(cov_tuned),
        "multi_cutoff_scalar_aiw_calib": float(aiw_tuned),
    }
    return tuned, summary


def build_split_artifacts(
    base_hist_df: pd.DataFrame,
    torvik: pd.DataFrame,
    train_end: int,
    calib_end: int,
    target_coverage: float,
) -> Optional[Dict[str, object]]:
    n_all = len(base_hist_df)
    if train_end <= 0 or calib_end <= train_end or calib_end >= n_all:
        return None

    model_df = base_hist_df.copy()
    train_cutoff_date = model_df.loc[train_end - 1, "date_dt"]
    calib_cutoff_date = model_df.loc[calib_end - 1, "date_dt"]

    model_df = apply_fixed_cutoff_snapshot(model_df, torvik, train_cutoff_date, SOURCE_STATS)
    model_df = add_features(model_df)
    model_df = model_df.dropna(subset=["margin"] + RIDGE_FEATURES + PRE_XGB_REQUIRED).copy()
    model_df = model_df.sort_values("date_dt").reset_index(drop=True)

    train_mask = model_df["date_dt"] <= train_cutoff_date
    calib_mask = (model_df["date_dt"] > train_cutoff_date) & (model_df["date_dt"] <= calib_cutoff_date)
    test_mask = model_df["date_dt"] > calib_cutoff_date
    if train_mask.sum() < 200 or calib_mask.sum() < 160 or test_mask.sum() < 80:
        return None

    model_df, ridge_model, ridge_scaler = fit_ridge_baseline(model_df, train_mask)
    model_df = model_df.dropna(subset=XGB_POINT_FEATURES).copy()
    model_df = model_df.sort_values("date_dt").reset_index(drop=True)

    train_mask = model_df["date_dt"] <= train_cutoff_date
    calib_mask = (model_df["date_dt"] > train_cutoff_date) & (model_df["date_dt"] <= calib_cutoff_date)
    test_mask = model_df["date_dt"] > calib_cutoff_date
    if train_mask.sum() < 200 or calib_mask.sum() < 160 or test_mask.sum() < 80:
        return None

    X_train = model_df.loc[train_mask, XGB_POINT_FEATURES].values
    y_train = model_df.loc[train_mask, "margin"].values
    xgb_model = fit_xgb_point_model(X_train, y_train)

    mu_all = dampen_point_predictions(xgb_model.predict(model_df[XGB_POINT_FEATURES].values))
    model_df["mu"] = mu_all

    calib_df = model_df.loc[calib_mask].reset_index(drop=True)
    test_df = model_df.loc[test_mask].reset_index(drop=True)
    y_calib = calib_df["margin"].values
    y_test = test_df["margin"].values
    mu_calib = calib_df["mu"].values
    mu_test = test_df["mu"].values

    allocator = fit_bucket_allocator(
        calib_df=calib_df,
        mu_calib=mu_calib,
        y_calib=y_calib,
        target_coverage=target_coverage,
    )

    return {
        "model_df": model_df,
        "train_cutoff_date": train_cutoff_date,
        "calib_cutoff_date": calib_cutoff_date,
        "train_mask": train_mask,
        "calib_mask": calib_mask,
        "test_mask": test_mask,
        "calib_df": calib_df,
        "test_df": test_df,
        "y_calib": y_calib,
        "y_test": y_test,
        "mu_calib": mu_calib,
        "mu_test": mu_test,
        "allocator": allocator,
        "ridge_model": ridge_model,
        "ridge_scaler": ridge_scaler,
        "xgb_model": xgb_model,
    }

def fit_bucket_allocator(
    calib_df: pd.DataFrame,
    mu_calib: np.ndarray,
    y_calib: np.ndarray,
    target_coverage: float,
) -> WidthAllocator:
    n = len(calib_df)
    fit_n = max(120, int(0.5 * n))
    if n - fit_n < 120:
        fit_n = max(80, n - 120)

    fit_df = calib_df.iloc[:fit_n].reset_index(drop=True)
    tune_df = calib_df.iloc[fit_n:].reset_index(drop=True)
    mu_fit = np.asarray(mu_calib[:fit_n], dtype=float)
    y_fit = np.asarray(y_calib[:fit_n], dtype=float)
    mu_tune = np.asarray(mu_calib[fit_n:], dtype=float)
    y_tune = np.asarray(y_calib[fit_n:], dtype=float)
    rounded_calib = ROUNDING_AWARE_CALIBRATION
    tune_folds = build_time_folds(len(tune_df))

    abs_err_fit = np.abs(y_fit - mu_fit)
    abs_err_tune = np.abs(y_tune - mu_tune)

    tempo_fit = pd.to_numeric(fit_df["tempo_avg"], errors="coerce").values
    tempo_fit = np.where(np.isfinite(tempo_fit), tempo_fit, np.nanmedian(tempo_fit[np.isfinite(tempo_fit)]))
    imbalance_fit = np.abs(pd.to_numeric(fit_df["imbalance_diff"], errors="coerce").values)
    imbalance_fit = np.where(
        np.isfinite(imbalance_fit),
        imbalance_fit,
        np.nanmedian(imbalance_fit[np.isfinite(imbalance_fit)]),
    )

    tempo_median = float(np.median(tempo_fit))
    imbalance_median = float(np.median(imbalance_fit))

    imb_lo_n = int((imbalance_fit <= imbalance_median).sum())
    imb_hi_n = int((imbalance_fit > imbalance_median).sum())
    use_imbalance = min(imb_lo_n, imb_hi_n) >= MIN_IMB_GROUP_N

    spread_code_fit, tempo_code_fit, imb_code_fit = row_bucket_codes(
        df=fit_df,
        mu=mu_fit,
        tempo_median=tempo_median,
        imbalance_median=imbalance_median,
    )
    full_p, st_p, s_p, global_p = create_profiles_by_level(
        spread_code=spread_code_fit,
        tempo_code=tempo_code_fit,
        imb_code=imb_code_fit,
        abs_res=abs_err_fit,
        use_imbalance=use_imbalance,
        max_cdf_t=MAX_CDF_T,
    )

    tempo_cut = tempo_median
    X_fit_u = build_uncertainty_matrix(fit_df, mu_fit, tempo_cut)
    X_tune_u = build_uncertainty_matrix(tune_df, mu_tune, tempo_cut)
    y_fit_u = np.log1p(abs_err_fit)

    uncertainty_scaler = StandardScaler()
    X_fit_u_s = uncertainty_scaler.fit_transform(X_fit_u)
    X_tune_u_s = uncertainty_scaler.transform(X_tune_u)

    uncertainty_model = Ridge(alpha=1.0)
    uncertainty_model.fit(X_fit_u_s, y_fit_u)

    raw_hw_tune = np.expm1(uncertainty_model.predict(X_tune_u_s))
    raw_hw_tune = np.clip(raw_hw_tune, 0.5, WIDTH_CAP)

    temp_allocator = WidthAllocator(
        use_imbalance=use_imbalance,
        tempo_median=tempo_median,
        imbalance_median=imbalance_median,
        tempo_cut=tempo_cut,
        uncertainty_scaler=uncertainty_scaler,
        uncertainty_model=uncertainty_model,
        base_c=1.0,
        calibration_mult=1.0,
        expected_target=target_coverage,
        target_coverage=target_coverage,
        width_cap=WIDTH_CAP,
        max_cdf_t=MAX_CDF_T,
        spread_post_multipliers=np.ones(4, dtype=float),
        pre_blowout_mult=1.0,
        tempo_low_mult=1.0,
        tempo_high_mult=1.0,
        blowout_cap=None,
        full_profiles=full_p,
        st_profiles=st_p,
        s_profiles=s_p,
        global_profile=global_p,
        calibration_summary={},
    )

    tune_profiles = map_profiles_for_rows(tune_df, mu_tune, temp_allocator)
    base_c, base_cov_tune, base_aiw_tune = choose_base_multiplier(
        mu=mu_tune,
        y_true=y_tune,
        base_hw=raw_hw_tune,
        target_coverage=target_coverage,
        width_cap=WIDTH_CAP,
        folds=tune_folds,
        rounded=rounded_calib,
    )
    base_w_tune = np.clip(raw_hw_tune * base_c, 0.5, WIDTH_CAP)

    et_low = max(0.70, target_coverage - 0.008)
    et_high = min(0.76, target_coverage + 0.002)
    candidates = np.round(np.arange(et_low, et_high + 1e-9, 0.001), 3)
    if len(candidates) == 0:
        candidates = np.array([target_coverage], dtype=float)

    best_tuple: Optional[Tuple[float, float, float, int, float]] = None
    for et in candidates:
        w_adj, _, added = greedy_allocate_width(
            base_widths=base_w_tune,
            profiles=tune_profiles,
            expected_target=float(et),
            width_cap=WIDTH_CAP,
            max_cdf_t=MAX_CDF_T,
            step=GREEDY_STEP,
            min_delta=MIN_DELTA,
        )
        cov, aiw, _, score = fold_calibration_score(
            mu=mu_tune,
            y_true=y_tune,
            widths=w_adj,
            target_coverage=target_coverage,
            folds=tune_folds,
            rounded=rounded_calib,
        )
        row = (float(et), cov, aiw, int(added), score)
        if best_tuple is None:
            best_tuple = row
            continue

        if row[4] < best_tuple[4] - 1e-12 or (abs(row[4] - best_tuple[4]) < 1e-12 and aiw < best_tuple[2]):
            best_tuple = row

    assert best_tuple is not None
    best_expected_target, greedy_cov_tune, greedy_aiw_tune, greedy_added_tune, greedy_score = best_tuple

    w_greedy_tune, _, _ = greedy_allocate_width(
        base_widths=base_w_tune,
        profiles=tune_profiles,
        expected_target=best_expected_target,
        width_cap=WIDTH_CAP,
        max_cdf_t=MAX_CDF_T,
        step=GREEDY_STEP,
        min_delta=MIN_DELTA,
    )
    mult_grid = np.arange(0.90, 1.16, 0.005)
    best_mult = 1.0
    best_mult_cov, best_mult_aiw, _, best_mult_score = fold_calibration_score(
        mu=mu_tune,
        y_true=y_tune,
        widths=w_greedy_tune,
        target_coverage=target_coverage,
        folds=tune_folds,
        rounded=rounded_calib,
    )
    for m in mult_grid:
        w_m = np.clip(w_greedy_tune * m, 0.5, WIDTH_CAP)
        cov_m, aiw_m, _, score_m = fold_calibration_score(
            mu=mu_tune,
            y_true=y_tune,
            widths=w_m,
            target_coverage=target_coverage,
            folds=tune_folds,
            rounded=rounded_calib,
        )
        if score_m < best_mult_score - 1e-12 or (abs(score_m - best_mult_score) < 1e-12 and aiw_m < best_mult_aiw):
            best_mult = float(m)
            best_mult_cov = cov_m
            best_mult_aiw = aiw_m
            best_mult_score = score_m

    w_cal = np.clip(w_greedy_tune * best_mult, 0.5, WIDTH_CAP)
    abs_mu_tune = np.abs(mu_tune)
    spread4 = audit_spread_bucket_index(abs_mu_tune)
    tempo_tune = pd.to_numeric(tune_df["tempo_avg"], errors="coerce").values
    tempo_tune = np.where(np.isfinite(tempo_tune), tempo_tune, tempo_median)
    low_tempo_mask = tempo_tune <= tempo_median
    high_tempo_mask = tempo_tune > tempo_median

    m0_grid = [1.00, 1.01]
    m1_grid = [0.98, 0.99, 1.00]
    m2_grid = [1.02, 1.03, 1.04, 1.05]
    m3_grid = [0.97, 0.98, 0.99, 1.00]
    low_grid = [1.00, 0.99]
    high_grid = [1.01, 1.02, 1.03]

    best_post = None
    for m0 in m0_grid:
        for m1 in m1_grid:
            for m2 in m2_grid:
                for m3 in m3_grid:
                    sm = np.array([m0, m1, m2, m3], dtype=float)
                    w_s = w_cal * sm[spread4]
                    for low_m in low_grid:
                        for high_m in high_grid:
                            w_adj = w_s.copy()
                            w_adj[low_tempo_mask] *= low_m
                            w_adj[high_tempo_mask] *= high_m
                            w_adj = np.clip(w_adj, 0.5, WIDTH_CAP)

                            cov_all, aiw_all, _, score_all = fold_calibration_score(
                                mu=mu_tune,
                                y_true=y_tune,
                                widths=w_adj,
                                target_coverage=target_coverage,
                                folds=tune_folds,
                                rounded=rounded_calib,
                            )
                            if cov_all < target_coverage:
                                continue

                            ok = True
                            for b in range(4):
                                mb = spread4 == b
                                if int(mb.sum()) >= 30:
                                    cov_b, _ = coverage_aiw_from_mu_width(
                                        mu=mu_tune[mb],
                                        y_true=y_tune[mb],
                                        widths=w_adj[mb],
                                        rounded=rounded_calib,
                                    )
                                    if cov_b < 0.70:
                                        ok = False
                                        break
                            if not ok:
                                continue

                            m_close = spread4 == 0
                            if int(m_close.sum()) >= 50:
                                cov_close, _ = coverage_aiw_from_mu_width(
                                    mu=mu_tune[m_close],
                                    y_true=y_tune[m_close],
                                    widths=w_adj[m_close],
                                    rounded=rounded_calib,
                                )
                                if cov_close < 0.75:
                                    continue

                            if int(high_tempo_mask.sum()) >= 50:
                                cov_high, _ = coverage_aiw_from_mu_width(
                                    mu=mu_tune[high_tempo_mask],
                                    y_true=y_tune[high_tempo_mask],
                                    widths=w_adj[high_tempo_mask],
                                    rounded=rounded_calib,
                                )
                                if cov_high < 0.70:
                                    continue

                            cov_low = (
                                coverage_aiw_from_mu_width(
                                    mu=mu_tune[low_tempo_mask],
                                    y_true=y_tune[low_tempo_mask],
                                    widths=w_adj[low_tempo_mask],
                                    rounded=rounded_calib,
                                )[0]
                                if int(low_tempo_mask.sum()) > 0
                                else cov_all
                            )
                            cov_high = (
                                coverage_aiw_from_mu_width(
                                    mu=mu_tune[high_tempo_mask],
                                    y_true=y_tune[high_tempo_mask],
                                    widths=w_adj[high_tempo_mask],
                                    rounded=rounded_calib,
                                )[0]
                                if int(high_tempo_mask.sum()) > 0
                                else cov_all
                            )
                            tempo_gap = abs(cov_low - cov_high)

                            cand = (score_all, aiw_all, tempo_gap, sm, low_m, high_m, cov_all, cov_low, cov_high)
                            if best_post is None or score_all < best_post[0] - 1e-12 or (
                                abs(score_all - best_post[0]) < 1e-12 and aiw_all < best_post[1] - 1e-12
                            ) or (
                                abs(score_all - best_post[0]) < 1e-12 and abs(aiw_all - best_post[1]) < 1e-12 and tempo_gap < best_post[2]
                            ):
                                best_post = cand

    if best_post is None:
        spread_post = np.ones(4, dtype=float)
        low_post = 1.0
        high_post = 1.0
        post_cov_tune, post_aiw_tune = coverage_aiw_from_mu_width(mu_tune, y_tune, w_cal, rounded=rounded_calib)
        post_cov_low = (
            coverage_aiw_from_mu_width(
                mu_tune[low_tempo_mask], y_tune[low_tempo_mask], w_cal[low_tempo_mask], rounded=rounded_calib
            )[0]
            if int(low_tempo_mask.sum()) > 0
            else post_cov_tune
        )
        post_cov_high = (
            coverage_aiw_from_mu_width(
                mu_tune[high_tempo_mask], y_tune[high_tempo_mask], w_cal[high_tempo_mask], rounded=rounded_calib
            )[0]
            if int(high_tempo_mask.sum()) > 0
            else post_cov_tune
        )
        post_score = abs(post_cov_tune - target_coverage) + (0.002 if post_cov_tune < target_coverage else 0.0)
    else:
        post_score, post_aiw_tune, _, spread_post, low_post, high_post, post_cov_tune, post_cov_low, post_cov_high = best_post

    spread_post = spread_post.copy()
    spread_post[0] = max(spread_post[0], 1.01)
    spread_post[2] = max(spread_post[2], 1.06)
    pre_blowout_mult = 1.00

    w_post = w_cal * spread_post[spread4]
    mid_mask = spread4 == 2
    w_post[mid_mask] *= pre_blowout_mult
    w_post[low_tempo_mask] *= low_post
    w_post[high_tempo_mask] *= high_post
    w_post = np.clip(w_post, 0.5, WIDTH_CAP)

    blow_mask = spread4 == 3
    blowout_candidates: List[Optional[float]] = [None, 29.0, 28.0]
    best_cap = None
    best_cap_cov, best_cap_aiw, _, best_cap_score = fold_calibration_score(
        mu=mu_tune,
        y_true=y_tune,
        widths=w_post,
        target_coverage=target_coverage,
        folds=tune_folds,
        rounded=rounded_calib,
    )
    best_cap_cov_blow = (
        coverage_aiw_from_mu_width(mu_tune[blow_mask], y_tune[blow_mask], w_post[blow_mask], rounded=rounded_calib)[0]
        if int(blow_mask.sum()) > 0
        else best_cap_cov
    )
    blow_cov_floor = 0.67

    for cap in blowout_candidates:
        w_cap = w_post.copy()
        if cap is not None and int(blow_mask.sum()) > 0:
            w_cap[blow_mask] = np.minimum(w_cap[blow_mask], cap / 2.0)
        w_cap = np.clip(w_cap, 0.5, WIDTH_CAP)

        cov_all, aiw_all, _, score_all = fold_calibration_score(
            mu=mu_tune,
            y_true=y_tune,
            widths=w_cap,
            target_coverage=target_coverage,
            folds=tune_folds,
            rounded=rounded_calib,
        )
        if cov_all < target_coverage:
            continue
        cov_blow = (
            coverage_aiw_from_mu_width(mu_tune[blow_mask], y_tune[blow_mask], w_cap[blow_mask], rounded=rounded_calib)[0]
            if int(blow_mask.sum()) > 0
            else cov_all
        )
        if int(blow_mask.sum()) > 0 and cov_blow < blow_cov_floor:
            continue

        if (
            score_all < best_cap_score - 1e-12
            or (abs(score_all - best_cap_score) < 1e-12 and aiw_all < best_cap_aiw - 1e-12)
        ):
            best_cap = cap
            best_cap_cov = cov_all
            best_cap_aiw = aiw_all
            best_cap_cov_blow = cov_blow
            best_cap_score = score_all

    return WidthAllocator(
        use_imbalance=use_imbalance,
        tempo_median=tempo_median,
        imbalance_median=imbalance_median,
        tempo_cut=tempo_cut,
        uncertainty_scaler=uncertainty_scaler,
        uncertainty_model=uncertainty_model,
        base_c=base_c,
        calibration_mult=best_mult,
        expected_target=best_expected_target,
        target_coverage=target_coverage,
        width_cap=WIDTH_CAP,
        max_cdf_t=MAX_CDF_T,
        spread_post_multipliers=spread_post,
        pre_blowout_mult=pre_blowout_mult,
        tempo_low_mult=float(low_post),
        tempo_high_mult=float(high_post),
        blowout_cap=best_cap,
        full_profiles=full_p,
        st_profiles=st_p,
        s_profiles=s_p,
        global_profile=global_p,
        calibration_summary={
            "fit_n": float(fit_n),
            "tune_n": float(len(tune_df)),
            "base_cov_tune": base_cov_tune,
            "base_aiw_tune": base_aiw_tune,
            "greedy_cov_tune": greedy_cov_tune,
            "greedy_aiw_tune": greedy_aiw_tune,
            "greedy_added_points_tune": float(greedy_added_tune) * GREEDY_STEP,
            "calibrated_cov_tune": best_mult_cov,
            "calibrated_aiw_tune": best_mult_aiw,
            "post_cov_tune": post_cov_tune,
            "post_aiw_tune": post_aiw_tune,
            "post_cov_low_tempo": post_cov_low,
            "post_cov_high_tempo": post_cov_high,
            "post_cov_tune_blowout_cap": best_cap_cov,
            "post_aiw_tune_blowout_cap": best_cap_aiw,
            "post_cov_blowout_cap_bucket": best_cap_cov_blow,
            "post_blowout_cap": float(best_cap) if best_cap is not None else -1.0,
            "post_pre_blowout_mult": pre_blowout_mult,
            "post_spread_m0": float(spread_post[0]),
            "post_spread_m1": float(spread_post[1]),
            "post_spread_m2": float(spread_post[2]),
            "post_spread_m3": float(spread_post[3]),
            "post_tempo_low_m": float(low_post),
            "post_tempo_high_m": float(high_post),
            "full_profile_count": float(len(full_p)),
            "st_profile_count": float(len(st_p)),
            "s_profile_count": float(len(s_p)),
            "calibration_rounded": 1.0 if rounded_calib else 0.0,
            "calibration_folds": float(len(tune_folds)),
            "greedy_score_tune": greedy_score,
            "mult_score_tune": best_mult_score,
            "post_score_tune": post_score,
            "cap_score_tune": best_cap_score,
        },
    )


def predict_half_widths(
    df: pd.DataFrame,
    mu: np.ndarray,
    allocator: WidthAllocator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    profiles = map_profiles_for_rows(df, mu, allocator)
    if allocator.uncertainty_scaler is not None and allocator.uncertainty_model is not None:
        X_u = build_uncertainty_matrix(df, mu, allocator.tempo_cut)
        X_u_s = allocator.uncertainty_scaler.transform(X_u)
        raw_hw = np.expm1(allocator.uncertainty_model.predict(X_u_s))
        raw_hw = np.clip(raw_hw, 0.5, allocator.width_cap)
    else:
        s = np.array([p.median_abs for p in profiles], dtype=float)
        raw_hw = s

    base_w = np.clip(allocator.base_c * raw_hw, 0.5, allocator.width_cap)
    _, exp_cov_base = expected_coverage_from_widths(base_w, profiles, allocator.max_cdf_t)

    final_w, exp_cov_final, added_points = greedy_allocate_width(
        base_widths=base_w,
        profiles=profiles,
        expected_target=allocator.expected_target,
        width_cap=allocator.width_cap,
        max_cdf_t=allocator.max_cdf_t,
        step=GREEDY_STEP,
        min_delta=MIN_DELTA,
    )
    final_w = np.clip(final_w * allocator.calibration_mult, 0.5, allocator.width_cap)
    abs_mu = np.abs(np.asarray(mu, dtype=float))
    spread4 = audit_spread_bucket_index(abs_mu)
    final_w = final_w * allocator.spread_post_multipliers[spread4]
    final_w[(abs_mu >= 12.0) & (abs_mu < 20.0)] *= allocator.pre_blowout_mult
    tempo = pd.to_numeric(df["tempo_avg"], errors="coerce").values
    tempo = np.where(np.isfinite(tempo), tempo, allocator.tempo_median)
    low_mask = tempo <= allocator.tempo_median
    high_mask = tempo > allocator.tempo_median
    final_w[low_mask] *= allocator.tempo_low_mult
    final_w[high_mask] *= allocator.tempo_high_mult
    if allocator.blowout_cap is not None:
        blow_mask = abs_mu >= 20.0
        final_w[blow_mask] = np.minimum(final_w[blow_mask], allocator.blowout_cap / 2.0)
    final_w = np.clip(final_w, 0.5, allocator.width_cap)
    _, exp_cov_final_cal = expected_coverage_from_widths(final_w, profiles, allocator.max_cdf_t)
    meta = {
        "expected_cov_base": exp_cov_base,
        "expected_cov_final": exp_cov_final,
        "expected_cov_final_cal": exp_cov_final_cal,
        "added_points": float(added_points) * GREEDY_STEP,
        "base_aiw": float(np.mean(2.0 * base_w)),
        "final_aiw": float(np.mean(2.0 * final_w)),
    }
    return final_w, meta


def evaluate_intervals(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rounded: bool = False,
) -> Tuple[float, float]:
    if rounded:
        lower, upper = round_interval_bounds(lower, upper)
    covered = (y_true >= lower) & (y_true <= upper)
    coverage = float(covered.mean())
    aiw = float(np.mean(upper - lower))
    return coverage, aiw


def prepare_historical_frame(all_games: pd.DataFrame, torvik: pd.DataFrame) -> pd.DataFrame:
    hist = all_games.copy()
    hist = hist[pd.to_numeric(hist["margin"], errors="coerce").notna()].copy()
    hist["margin"] = pd.to_numeric(hist["margin"], errors="coerce")
    hist = hist.dropna(subset=["margin"]).copy()
    merged = merge_torvik_asof(hist, torvik, SOURCE_STATS)
    feat = add_features(merged)
    feat = feat.sort_values("date_dt").reset_index(drop=True)
    return feat


def run_backtest(target_coverage: float = 0.74) -> Dict[str, object]:
    all_games, torvik, _ = load_data()
    base_hist_df = prepare_historical_frame(all_games, torvik)

    n_all = len(base_hist_df)
    train_end = int(0.6 * n_all)
    calib_end = int(0.8 * n_all)
    split_art = build_split_artifacts(
        base_hist_df=base_hist_df,
        torvik=torvik,
        train_end=train_end,
        calib_end=calib_end,
        target_coverage=target_coverage,
    )
    if split_art is None:
        raise RuntimeError("Final split has too few rows after feature filtering. Check data availability.")

    model_df = split_art["model_df"]
    train_cutoff_date = split_art["train_cutoff_date"]
    calib_cutoff_date = split_art["calib_cutoff_date"]
    train_mask = split_art["train_mask"]
    calib_mask = split_art["calib_mask"]
    test_mask = split_art["test_mask"]
    calib_df = split_art["calib_df"]
    test_df = split_art["test_df"]
    y_calib = split_art["y_calib"]
    y_test = split_art["y_test"]
    mu_calib = split_art["mu_calib"]
    mu_test = split_art["mu_test"]
    ridge_model = split_art["ridge_model"]
    ridge_scaler = split_art["ridge_scaler"]
    xgb_model = split_art["xgb_model"]
    allocator = split_art["allocator"]

    multi_cutoff_used = 0
    if USE_MULTI_CUTOFF_STABILITY:
        stable_allocs: List[WidthAllocator] = []
        for tr_frac, cal_frac in MULTI_CUTOFF_FRACTIONS:
            tr_end = int(tr_frac * n_all)
            ca_end = int(cal_frac * n_all)
            if ca_end >= calib_end:
                continue
            sub = build_split_artifacts(
                base_hist_df=base_hist_df,
                torvik=torvik,
                train_end=tr_end,
                calib_end=ca_end,
                target_coverage=target_coverage,
            )
            if sub is not None:
                stable_allocs.append(sub["allocator"])

        if len(stable_allocs) > 0:
            multi_cutoff_used = len(stable_allocs)
            allocator = aggregate_allocators(stable_allocs, allocator)
            allocator, scalar_summary = tune_allocator_scalar_on_calib(
                allocator=allocator,
                calib_df=calib_df,
                mu_calib=mu_calib,
                y_calib=y_calib,
                target_coverage=target_coverage,
            )
            allocator.calibration_summary.update(scalar_summary)
            allocator.calibration_summary["multi_cutoff_used"] = float(multi_cutoff_used)
        else:
            allocator.calibration_summary["multi_cutoff_used"] = 0.0

    hw_test, meta_test = predict_half_widths(test_df, mu_test, allocator)
    lower_test = mu_test - hw_test
    upper_test = mu_test + hw_test
    test_cov_raw, test_aiw_raw = evaluate_intervals(y_test, lower_test, upper_test, rounded=False)
    test_cov, test_aiw = evaluate_intervals(
        y_test,
        lower_test,
        upper_test,
        rounded=ROUNDING_AWARE_CALIBRATION,
    )

    point_mae_train = mean_absolute_error(model_df.loc[train_mask, "margin"], model_df.loc[train_mask, "mu"])
    point_mae_calib = mean_absolute_error(y_calib, mu_calib)
    point_mae_test = mean_absolute_error(y_test, mu_test)

    print("=" * 76)
    print("PREDICTION INTERVAL BACKTEST (BUCKETED SCALE + GREEDY ALLOCATION)")
    print("=" * 76)
    print(f"rows: total={len(model_df)} train={train_mask.sum()} calib={calib_mask.sum()} test={test_mask.sum()}")
    print(f"train cutoff: {train_cutoff_date.date()} | calib cutoff: {calib_cutoff_date.date()}")
    print(f"point MAE: train={point_mae_train:.3f} calib={point_mae_calib:.3f} test={point_mae_test:.3f}")
    print(
        f"bucket model: spread({NUM_SPREAD_BUCKETS}) x tempo(2) x imbalance({'2' if allocator.use_imbalance else '1'}) "
        f"| base_mult={allocator.base_c:.3f}"
    )
    print(
        f"calib tune: base_cov={allocator.calibration_summary['base_cov_tune']:.1%} "
        f"base_aiw={allocator.calibration_summary['base_aiw_tune']:.3f} "
        f"greedy_cov={allocator.calibration_summary['greedy_cov_tune']:.1%} "
        f"greedy_aiw={allocator.calibration_summary['greedy_aiw_tune']:.3f} "
        f"cal_cov={allocator.calibration_summary['calibrated_cov_tune']:.1%} "
        f"cal_aiw={allocator.calibration_summary['calibrated_aiw_tune']:.3f} "
        f"post_cov={allocator.calibration_summary['post_cov_tune']:.1%} "
        f"post_aiw={allocator.calibration_summary['post_aiw_tune']:.3f} "
        f"exp_target={allocator.expected_target:.3f} mult={allocator.calibration_mult:.2f}"
    )
    print(
        f"calib mode: rounded={bool(allocator.calibration_summary['calibration_rounded'])} "
        f"folds={int(allocator.calibration_summary['calibration_folds'])}"
    )
    print(
        f"stability mode: multi_cutoff={bool(multi_cutoff_used)} "
        f"used={int(allocator.calibration_summary.get('multi_cutoff_used', 0.0))} "
        f"scalar_m={allocator.calibration_summary.get('multi_cutoff_scalar_m', 1.0):.3f}"
    )
    print(
        "post factors: "
        f"spread[<5={allocator.calibration_summary['post_spread_m0']:.2f}, "
        f"5-12={allocator.calibration_summary['post_spread_m1']:.2f}, "
        f"12-20={allocator.calibration_summary['post_spread_m2']:.2f}, "
        f"20+={allocator.calibration_summary['post_spread_m3']:.2f}] "
        f"pre_blowout_12_20={allocator.calibration_summary['post_pre_blowout_mult']:.2f} "
        f"tempo[low={allocator.calibration_summary['post_tempo_low_m']:.2f}, "
        f"high={allocator.calibration_summary['post_tempo_high_m']:.2f}] "
        f"blowout_cap={allocator.calibration_summary['post_blowout_cap']:.0f}"
    )
    print(
        f"test expected: base_cov={meta_test['expected_cov_base']:.1%} "
        f"final_cov={meta_test['expected_cov_final']:.1%} "
        f"final_cov_cal={meta_test['expected_cov_final_cal']:.1%} "
        f"added_points={int(meta_test['added_points'])}"
    )
    print(
        f"PI target={target_coverage:.1%} | test coverage={test_cov:.1%} | test AIW={test_aiw:.3f} "
        f"(raw: cov={test_cov_raw:.1%}, aiw={test_aiw_raw:.3f})"
    )
    if test_cov >= 0.70:
        print("status: meets 70% competition threshold")
    else:
        print("status: below 70% competition threshold")

    return {
        "model_df": model_df,
        "train_cutoff_date": train_cutoff_date,
        "calib_cutoff_date": calib_cutoff_date,
        "ridge_model": ridge_model,
        "ridge_scaler": ridge_scaler,
        "xgb_model": xgb_model,
        "allocator": allocator,
        "test_coverage": test_cov,
        "test_aiw": test_aiw,
    }

def predict_future_games(target_coverage: float = 0.74) -> pd.DataFrame:
    artifacts = run_backtest(target_coverage=target_coverage)
    allocator: WidthAllocator = artifacts["allocator"]

    all_games, torvik, future_games = load_data()
    if len(future_games) == 0:
        raise RuntimeError("No rows found in data/future_acc_games.csv")

    hist_df = prepare_historical_frame(all_games, torvik)
    hist_df = hist_df.dropna(subset=["margin"] + RIDGE_FEATURES + PRE_XGB_REQUIRED).copy()
    hist_df = hist_df.sort_values("date_dt").reset_index(drop=True)
    all_train_mask = pd.Series(True, index=hist_df.index)

    hist_df, ridge_model, ridge_scaler = fit_ridge_baseline(hist_df, all_train_mask)
    hist_df = hist_df.dropna(subset=XGB_POINT_FEATURES).copy()
    hist_df = hist_df.sort_values("date_dt").reset_index(drop=True)

    xgb_model = fit_xgb_point_model(
        hist_df[XGB_POINT_FEATURES].values,
        hist_df["margin"].values,
    )

    future_feat = merge_torvik_asof(future_games.copy(), torvik, SOURCE_STATS)
    future_schedule = pd.concat(
        [
            all_games[["date_dt", "home_team_id", "away_team_id"]],
            future_games[["date_dt", "home_team_id", "away_team_id"]],
        ],
        ignore_index=True,
    )
    future_feat = overwrite_rest_from_schedule(future_feat, future_schedule)
    future_feat = add_features(future_feat)
    future_feat = future_feat.dropna(subset=RIDGE_FEATURES + PRE_XGB_REQUIRED).copy()
    future_feat = future_feat.sort_values("date_dt").reset_index(drop=True)

    X_future_ridge = future_feat[RIDGE_FEATURES].values
    future_feat["ridge_pred_spread"] = ridge_model.predict(ridge_scaler.transform(X_future_ridge))
    future_feat = future_feat.dropna(subset=XGB_POINT_FEATURES).copy()

    mu = xgb_model.predict(future_feat[XGB_POINT_FEATURES].values)
    mu = dampen_point_predictions(mu)
    hw, meta_future = predict_half_widths(future_feat, mu, allocator)
    lower = mu - hw
    upper = mu + hw

    out = future_feat.copy()
    out["predicted_margin"] = mu
    out["lower_bound"] = lower
    out["upper_bound"] = upper
    out["interval_width"] = upper - lower

    cols = [
        "event_id",
        "date",
        "home_team_id",
        "away_team_id",
        "predicted_margin",
        "lower_bound",
        "upper_bound",
        "interval_width",
    ]
    out[cols].to_csv(OUTPUT_PATH, index=False)
    print(
        f"saved: {OUTPUT_PATH} ({len(out)} games) "
        f"| exp_cov_base={meta_future['expected_cov_base']:.1%} "
        f"exp_cov_final={meta_future['expected_cov_final']:.1%} "
        f"exp_cov_final_cal={meta_future['expected_cov_final_cal']:.1%}"
    )
    return out[cols]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-coverage", type=float, default=0.74)
    parser.add_argument("--predict-future", action="store_true")
    args = parser.parse_args()

    if args.predict_future:
        predict_future_games(target_coverage=args.target_coverage)
    else:
        run_backtest(target_coverage=args.target_coverage)


if __name__ == "__main__":
    main()
