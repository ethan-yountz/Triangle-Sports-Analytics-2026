"""Train and evaluate a fixed-cutoff XGBoost spread model."""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .ridge_model import RIDGE_FINAL_FEATURES


TORVIK_PATH = os.path.join("data", "torvik_asof_ratings_all_teams.csv")
ALL_GAMES_PATH = os.path.join("data", "all_games.csv")
ACC_TEAMS_PATH = os.path.join("data", "acc_teams.csv")
TEST_START_DATE = pd.Timestamp("2026-02-01")


def load_data():
    acc_teams = pd.read_csv(ACC_TEAMS_PATH)
    torvik = pd.read_csv(TORVIK_PATH)
    all_games = pd.read_csv(ALL_GAMES_PATH)

    torvik["team_id"] = pd.to_numeric(torvik["team_id"], errors="coerce")
    torvik = torvik[torvik["team_id"].notna()].copy()
    torvik["team_id"] = torvik["team_id"].astype(int)
    return acc_teams, torvik, all_games


def merge_torvik_asof(games: pd.DataFrame, torvik: pd.DataFrame, stat_cols: List[str]) -> pd.DataFrame:
    games = games.copy()
    games["date"] = pd.to_datetime(games["date"]).dt.strftime("%Y-%m-%d")
    games["as_of_date"] = (
        pd.to_datetime(games["date"]) - pd.Timedelta(days=1)
    ).dt.strftime("%Y-%m-%d")

    torvik = torvik.copy()
    torvik["date"] = pd.to_datetime(torvik["date"]).dt.strftime("%Y-%m-%d")

    home_cols = {c: f"home_{c}" for c in stat_cols}
    away_cols = {c: f"away_{c}" for c in stat_cols}

    home = torvik[["team_id", "date"] + stat_cols].rename(columns=home_cols)
    home = home.rename(columns={"team_id": "home_team_id", "date": "as_of_date"})
    away = torvik[["team_id", "date"] + stat_cols].rename(columns=away_cols)
    away = away.rename(columns={"team_id": "away_team_id", "date": "as_of_date"})

    out = games.merge(home, on=["home_team_id", "as_of_date"], how="left")
    out = out.merge(away, on=["away_team_id", "as_of_date"], how="left")
    return out


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

    out["adj_tempo_diff"] = out["home_adj_tempo"] - out["away_adj_tempo"]
    out["adj_tempo_avg"] = (out["home_adj_tempo"] + out["away_adj_tempo"]) / 2.0
    out["tempo_avg"] = out["adj_tempo_avg"]
    out["tempo_x_adj_o_diff"] = out["adj_tempo_avg"] * out["adj_o_diff"]
    out["tempo_x_adj_d_diff"] = out["adj_tempo_avg"] * out["adj_d_diff"]
    out["tempo_x_efg_diff"] = out["adj_tempo_avg"] * out["efg_diff"]

    out["three_pt_rate_diff"] = out["home_three_pt_rate"] - out["away_three_pt_rate"]
    out["three_pt_pct_diff"] = out["home_three_pt_pct"] - out["away_three_pt_pct"]
    out["two_pt_pct_diff"] = out["home_two_pt_pct"] - out["away_two_pt_pct"]
    out["efgd_diff"] = out["home_efgd"] - out["away_efgd"]
    out["tempo_x_efgd_diff"] = out["adj_tempo_avg"] * out["efgd_diff"]

    out["three_pt_rate_avg"] = (out["home_three_pt_rate"] + out["away_three_pt_rate"]) / 2.0
    out["three_pt_pct_avg"] = (out["home_three_pt_pct"] + out["away_three_pt_pct"]) / 2.0
    out["var_proxy_tempo_3pr"] = out["three_pt_rate_avg"] * out["adj_tempo_avg"]
    out["var_proxy_3pr_1m3pp"] = out["three_pt_rate_avg"] * (1.0 - out["three_pt_pct_avg"] / 100.0)

    out["abs_adj_o_diff"] = out["adj_o_diff"].abs()
    out["abs_adj_d_diff"] = out["adj_d_diff"].abs()
    out["abs_efg_diff"] = out["efg_diff"].abs()
    out["od_imbalance_home"] = (out["home_adj_o"] - out["home_adj_d"]).abs()
    out["od_imbalance_away"] = (out["away_adj_o"] - out["away_adj_d"]).abs()
    out["imbalance_diff"] = out["od_imbalance_home"] - out["od_imbalance_away"]

    out["barthag_avg"] = (out["home_barthag"] + out["away_barthag"]) / 2.0
    out["wab_avg"] = (out["home_wab"] + out["away_wab"]) / 2.0

    out["home_days_since_last_game"] = pd.to_numeric(out["home_days_since_last_game"], errors="coerce")
    out["away_days_since_last_game"] = pd.to_numeric(out["away_days_since_last_game"], errors="coerce")
    out["home_games_last_7_days"] = pd.to_numeric(out["home_games_last_7_days"], errors="coerce")
    out["away_games_last_7_days"] = pd.to_numeric(out["away_games_last_7_days"], errors="coerce")

    out["days_since_last_game_diff"] = (
        out["home_days_since_last_game"] - out["away_days_since_last_game"]
    )
    out["games_last_7_days_diff"] = (
        out["home_games_last_7_days"] - out["away_games_last_7_days"]
    )
    out["home_fatigue_index"] = (
        out["home_games_last_7_days"] / np.maximum(out["home_days_since_last_game"], 1.0)
    )
    out["away_fatigue_index"] = (
        out["away_games_last_7_days"] / np.maximum(out["away_days_since_last_game"], 1.0)
    )
    out["fatigue_index_diff"] = out["home_fatigue_index"] - out["away_fatigue_index"]

    out["home_ftr"] = pd.to_numeric(out["home_ftr"], errors="coerce")
    out["away_ftr"] = pd.to_numeric(out["away_ftr"], errors="coerce")
    out["home_ftrd"] = pd.to_numeric(out["home_ftrd"], errors="coerce")
    out["away_ftrd"] = pd.to_numeric(out["away_ftrd"], errors="coerce")
    out["ftr_diff"] = out["home_ftr"] - out["away_ftr"]
    out["ftr_allowed_diff"] = out["home_ftrd"] - out["away_ftrd"]
    out["home_foul_pressure"] = out["home_ftr"] - out["away_ftrd"]
    out["away_foul_pressure"] = out["away_ftr"] - out["home_ftrd"]
    out["foul_rate_diff"] = out["home_foul_pressure"] - out["away_foul_pressure"]
    out["opponent_foul_rate_diff"] = out["away_ftrd"] - out["home_ftrd"]
    out["close_spread_proxy"] = 1.0 / (1.0 + (out["adj_o_diff"] - out["adj_d_diff"]).abs())
    out["ftr_x_close_spread"] = out["ftr_diff"] * out["close_spread_proxy"]
    out["ftr_x_fatigue"] = out["ftr_diff"] * out["fatigue_index_diff"]

    return out


def overwrite_rest_from_schedule(
    target_games: pd.DataFrame,
    schedule_games: pd.DataFrame,
    update_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Overwrite rest/congestion fields from schedule history."""
    out = target_games.copy()
    if update_mask is None:
        update_mask = pd.Series(True, index=out.index)
    else:
        update_mask = update_mask.reindex(out.index).fillna(False).astype(bool)
    if not update_mask.any():
        return out

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

    for idx, row in out.loc[update_mask].iterrows():
        game_date = pd.to_datetime(row["date_dt"]).to_datetime64()
        for side in ("home", "away"):
            team_id = int(row[f"{side}_team_id"])
            dates = team_dates.get(team_id, np.array([], dtype="datetime64[ns]"))
            pos = int(np.searchsorted(dates, game_date, side="left"))

            if pos == 0:
                days = 1.0
            else:
                days = float((game_date - dates[pos - 1]) / np.timedelta64(1, "D"))

            window_start = game_date - np.timedelta64(7, "D")
            lo = int(np.searchsorted(dates, window_start, side="left"))
            games7 = float(pos - lo)
            if games7 <= 0:
                games7 = 1.0

            out.loc[idx, f"{side}_days_since_last_game"] = days
            out.loc[idx, f"{side}_games_last_7_days"] = games7

    return out


def apply_fixed_cutoff_snapshot(
    df: pd.DataFrame, torvik: pd.DataFrame, cutoff_date: pd.Timestamp, stat_cols: List[str]
) -> pd.DataFrame:
    out = df.copy()
    future_mask = out["date_dt"] > cutoff_date
    if not future_mask.any():
        return out

    torvik_cut = torvik.copy()
    torvik_cut["date_dt"] = pd.to_datetime(torvik_cut["date"])
    torvik_cut = torvik_cut[torvik_cut["date_dt"] <= cutoff_date]
    torvik_cut = torvik_cut.sort_values("date_dt").groupby("team_id", as_index=False).tail(1)

    for stat in stat_cols:
        snapshot = torvik_cut.set_index("team_id")[stat].to_dict()
        out.loc[future_mask, f"home_{stat}"] = out.loc[future_mask, "home_team_id"].map(snapshot)
        out.loc[future_mask, f"away_{stat}"] = out.loc[future_mask, "away_team_id"].map(snapshot)
    return out


def build_ridge_baseline_feature(
    df: pd.DataFrame, train_mask: pd.Series, test_mask: pd.Series
) -> pd.DataFrame:
    out = df.copy()
    ridge_feats = list(RIDGE_FINAL_FEATURES)
    out = out.dropna(subset=ridge_feats + ["margin"]).copy()

    X_train = out.loc[train_mask.loc[out.index], ridge_feats].values
    y_train = out.loc[train_mask.loc[out.index], "margin"].values
    X_test = out.loc[test_mask.loc[out.index], ridge_feats].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=TimeSeriesSplit(n_splits=5))
    ridge_cv.fit(X_train_s, y_train)
    ridge = Ridge(alpha=float(ridge_cv.alpha_))
    ridge.fit(X_train_s, y_train)

    out.loc[train_mask.loc[out.index], "ridge_pred_spread"] = ridge.predict(X_train_s)
    out.loc[test_mask.loc[out.index], "ridge_pred_spread"] = ridge.predict(X_test_s)
    return out


def add_ridge_dependent_xgb_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ftr = pd.to_numeric(out.get("ftr_diff"), errors="coerce")
    ridge_base = np.abs(pd.to_numeric(out.get("ridge_pred_spread"), errors="coerce"))
    out["ftr_x_ridge_pred"] = ftr * ridge_base
    return out


def main():
    acc_teams, torvik, all_games = load_data()
    _ = acc_teams

    torvik_ids = set(torvik["team_id"].astype(int).tolist())
    games = all_games.copy()
    games["home_team_id"] = games["home_team_id"].astype(int)
    games["away_team_id"] = games["away_team_id"].astype(int)
    games = games[
        games["home_team_id"].isin(torvik_ids) & games["away_team_id"].isin(torvik_ids)
    ].reset_index(drop=True)

    source_stats = [
        "adj_o",
        "adj_d",
        "efg",
        "efgd",
        "ftr",
        "ftrd",
        "tor",
        "tord",
        "orb",
        "drb",
        "three_pt_rate",
        "three_pt_pct",
        "two_pt_pct",
        "adj_tempo",
        "barthag",
        "wab",
        "days_since_last_game",
        "games_last_7_days",
    ]

    merged = merge_torvik_asof(games, torvik, source_stats)
    merged = add_features(merged)

    model_df = merged.copy()
    model_df["date_dt"] = pd.to_datetime(model_df["date"], errors="coerce")
    model_df = model_df.dropna(subset=["date_dt", "margin"]).copy()
    model_df = model_df.sort_values("date_dt").reset_index(drop=True)

    cutoff_date = TEST_START_DATE - pd.Timedelta(days=1)
    train_mask = model_df["date_dt"] < TEST_START_DATE
    test_mask = model_df["date_dt"] >= TEST_START_DATE

    model_df = apply_fixed_cutoff_snapshot(model_df, torvik, cutoff_date, source_stats)
    schedule_df = games[["date", "home_team_id", "away_team_id"]].copy()
    schedule_df["date_dt"] = pd.to_datetime(schedule_df["date"], errors="coerce")
    model_df = overwrite_rest_from_schedule(model_df, schedule_df, update_mask=test_mask)
    model_df = add_features(model_df)

    model_df = build_ridge_baseline_feature(model_df, train_mask, test_mask)
    model_df = add_ridge_dependent_xgb_features(model_df)

    xgb_features = [
        "adj_o_diff",
        "adj_d_diff",
        "tempo_x_adj_o_diff",
        "tempo_x_adj_d_diff",
        "tempo_x_efg_diff",
        "tempo_x_efgd_diff",
        "days_since_last_game_diff",
        "games_last_7_days_diff",
        "fatigue_index_diff",
        "adj_tempo_avg",
        "ftr_diff",
        "ftr_allowed_diff",
        "foul_rate_diff",
        "opponent_foul_rate_diff",
        "ftr_x_ridge_pred",
        "ftr_x_fatigue",
    ]
    model_df = model_df.dropna(subset=["margin", "ridge_pred_spread"] + xgb_features).copy()
    model_df = model_df.sort_values("date_dt").reset_index(drop=True)
    train_mask = model_df["date_dt"] < TEST_START_DATE
    test_mask = model_df["date_dt"] >= TEST_START_DATE

    X_train = model_df.loc[train_mask, xgb_features].values
    y_train = model_df.loc[train_mask, "margin"].values
    base_train = pd.to_numeric(model_df.loc[train_mask, "ridge_pred_spread"], errors="coerce").values
    X_test = model_df.loc[test_mask, xgb_features].values
    y_test = model_df.loc[test_mask, "margin"].values
    base_test = pd.to_numeric(model_df.loc[test_mask, "ridge_pred_spread"], errors="coerce").values

    train_n = len(X_train)
    val_n = max(200, int(train_n * 0.15))
    if val_n >= train_n:
        val_n = max(20, train_n // 5)
    val_n = min(val_n, train_n - 50)
    tr_end = train_n - val_n
    if tr_end < 50 or val_n < 10:
        raise ValueError("Insufficient rows for XGB inner validation split.")

    X_tr, y_tr = X_train[:tr_end], y_train[:tr_end]
    base_tr = base_train[:tr_end]
    X_val, y_val = X_train[tr_end:], y_train[tr_end:]
    base_val = base_train[tr_end:]

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
    xgb_early = XGBRegressor(
        **base_params,
        n_estimators=4000,
        early_stopping_rounds=200,
    )
    xgb_early.fit(
        X_tr,
        y_tr,
        base_margin=base_tr,
        eval_set=[(X_val, y_val)],
        base_margin_eval_set=[base_val],
        verbose=False,
    )

    best_iter_raw = getattr(xgb_early, "best_iteration", None)
    best_iter = int(best_iter_raw) if best_iter_raw is not None else 3999
    xgb = XGBRegressor(
        **base_params,
        n_estimators=best_iter + 1,
    )
    xgb.fit(X_train, y_train, base_margin=base_train, verbose=False)
    y_test_pred = xgb.predict(X_test, base_margin=base_test)

    test_df = model_df.loc[test_mask].copy().reset_index(drop=True)
    test_df["pred_xgb"] = y_test_pred

    spread_mask = test_df["spread"].notna()
    vegas_pred = -test_df.loc[spread_mask, "spread"]
    vegas_true = test_df.loc[spread_mask, "margin"]

    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    vegas_mae = mean_absolute_error(vegas_true, vegas_pred)
    close_mask = spread_mask & (test_df["spread"].abs() < 5)
    close_mae = mean_absolute_error(
        test_df.loc[close_mask, "margin"], test_df.loc[close_mask, "pred_xgb"]
    )
    model_bias = (test_df.loc[spread_mask, "pred_xgb"] - vegas_true).mean()
    vegas_bias = (vegas_pred - vegas_true).mean()

    print("XGBoost Fixed-Cutoff Results")
    print(f"cutoff_date: {cutoff_date.date()}")
    print(f"train_n: {int(train_mask.sum())}, test_n: {int(test_mask.sum())}")
    print(f"features: {xgb_features}")
    print(f"val_n_for_early_stopping: {val_n}")
    print(f"best_iteration: {best_iter if best_iter is not None else 'full_model'}")
    print(f"test_mae: {test_mae:.3f}")
    print(f"test_rmse: {test_rmse:.3f}")
    print(f"vegas_test_mae: {vegas_mae:.3f}")
    print(f"close_game_mae_|spread|<5: {close_mae:.3f} (n={int(close_mask.sum())})")
    print(f"model_bias_pred_minus_actual: {model_bias:+.3f}")
    print(f"vegas_bias_pred_minus_actual: {vegas_bias:+.3f}")
    print(f"mae_delta_model_minus_vegas: {test_mae - vegas_mae:+.3f}")


if __name__ == "__main__":
    main()
