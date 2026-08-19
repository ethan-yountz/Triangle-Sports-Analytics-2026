"""Walk-forward ridge evaluation with fixed outer folds and train-only preprocessing."""

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=PerformanceWarning)


ACC_TEAMS_PATH = os.path.join("data", "acc_teams.csv")
ALL_GAMES_PATH = os.path.join("data", "all_games.csv")
TORVIK_ALL_PATH = os.path.join("data", "torvik_asof_ratings_all_teams.csv")
TORVIK_FALLBACK_PATH = os.path.join("data", "torvik_asof_ratings.csv")

BASELINE_FEATURES = ["adj_o_diff", "adj_d_diff", "is_neutral"]
BASE_FEATURES: List[str] = []
FINAL_BLOCKS = [6, 9, 10, 11]
FEATURE_BLOCKS: List[Tuple[str, List[str]]] = [
    (
        "Block 1 Shooting Efficiency",
        [
            "efgd_diff",
            "two_pt_pct_diff",
            "two_pt_pct_d_diff",
            "three_pt_pct_diff",
            "three_pt_pct_d_diff",
            "three_pt_rate_diff",
            "three_pt_rate_d_diff",
        ],
    ),
    ("Block 2 Possession Control", ["tor_diff", "tord_diff", "orb_diff", "drb_diff"]),
    (
        "Block 3 Pace/Style",
        [
            "adj_tempo_diff",
            "tempo_x_tor_diff",
            "tempo_x_tord_diff",
            "tempo_x_orb_diff",
            "tempo_x_drb_diff",
        ],
    ),
    (
        "Block 4 Context/Fatigue",
        [
            "rest_diff",
            "home_is_b2b",
            "away_is_b2b",
            "b2b_diff",
            "home_short_rest",
            "away_short_rest",
            "short_rest_diff",
            "home_long_rest",
            "away_long_rest",
            "long_rest_diff",
        ],
    ),
    (
        "Block 5 Shot-Profile Asymmetry",
        [
            "three_pt_style_mismatch_diff",
            "two_pt_style_mismatch_diff",
            "shot_profile_asymmetry",
            "rate_x_pct_mismatch",
        ],
    ),
    (
        "Block 6 Tempo x Efficiency",
        ["tempo_x_adj_o_diff", "tempo_x_adj_d_diff", "tempo_x_efg_diff", "tempo_x_efgd_diff"],
    ),
    (
        "Block 7 Rest & Congestion",
        ["days_since_last_game_avg", "games_last_7_days_avg", "fatigue_index_avg"],
    ),
    (
        "Block 8 Variance Proxy",
        ["var_proxy_tempo_3pr", "var_proxy_3pr_1m3pp", "var_proxy_tempo_3pp_gap"],
    ),
    (
        "Block 9 Fatigue Asymmetry",
        ["days_since_last_game_diff", "games_last_7_days_diff", "fatigue_index_diff"],
    ),
    ("Block 10 Game Tempo Average", ["adj_tempo_avg"]),
    (
        "Block 11 Free Throw Leverage / Foul Ecology",
        [
            "ftr_diff",
            "ftr_allowed_diff",
            "foul_rate_diff",
            "opponent_foul_rate_diff",
            "ftr_x_close_spread",
            "ftr_x_fatigue",
        ],
    ),
    (
        "Block 12 Luck / Regression Signal",
        [
            "luck_diff",
            "recent_luck_trend_diff",
            "luck_x_recent_form",
            "luck_reversion_term",
            "luck_x_tempo",
        ],
    ),
    ("Block 13 eFG Diff", ["efg_diff"]),
    (
        "Block 14 Rolling Form RF1 EWMA-5",
        [
            "adj_o_ewm5_diff",
            "adj_d_ewm5_diff",
            "barthag_ewm5_diff",
            "efg_ewm5_diff",
            "efgd_ewm5_diff",
            "tor_ewm5_diff",
            "tord_ewm5_diff",
            "adj_tempo_ewm5_diff",
        ],
    ),
    (
        "Block 15 Rolling Form RF2 Delta-14",
        [
            "delta14_adj_o_diff",
            "delta14_adj_d_diff",
            "delta14_barthag_diff",
            "delta14_efg_diff",
            "delta14_tor_diff",
            "delta14_wab_diff",
        ],
    ),
    (
        "Block 16 Rolling Form RF3 Std-14",
        ["std14_adj_o_avg", "std14_adj_d_avg", "std14_efg_avg", "std14_tempo_avg"],
    ),
    (
        "Block 17 Recent Opponent Quality",
        [
            "opp_avg_barthag_last5_diff",
            "opp_avg_adj_o_last5_diff",
            "opp_avg_adj_d_last5_diff",
        ],
    ),
    (
        "Block 18 Venue Splits",
        ["home_home_net_rating", "away_away_net_rating", "venue_split_edge"],
    ),
    (
        "Block 19 TO/Reb Mismatch Interactions",
        ["tor_mismatch", "orb_mismatch"],
    ),
]


def unique_in_order(values: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def block_id_to_features() -> Dict[int, List[str]]:
    mapping: Dict[int, List[str]] = {}
    for idx, (_, feats) in enumerate(FEATURE_BLOCKS, start=1):
        mapping[idx] = list(feats)
    missing = [block_id for block_id in FINAL_BLOCKS if block_id not in mapping]
    if missing:
        raise ValueError(f"Missing FINAL_BLOCKS in FEATURE_BLOCKS: {missing}")
    return mapping


def build_final_feature_list() -> List[str]:
    mapping = block_id_to_features()
    feats = list(BASELINE_FEATURES)
    for block_id in FINAL_BLOCKS:
        feats.extend(mapping[block_id])
    return unique_in_order(feats)


RIDGE_FINAL_FEATURES = build_final_feature_list()
# Keep legacy name wired to the final ridge production feature set.
BASE_FEATURES = list(RIDGE_FINAL_FEATURES)

ALPHA_GRID = np.logspace(-3, 3, 50)
P_VALUE_PERMUTATIONS = 5000
MIN_TRAIN_ROWS = 75
MIN_TEST_ROWS = 5
RANDOM_SEED = 42


@dataclass(frozen=True)
class FoldSpec:
    name: str
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def train_end(self) -> pd.Timestamp:
        return self.test_start

    @property
    def snapshot_cutoff(self) -> pd.Timestamp:
        return self.train_end - pd.Timedelta(days=1)


OUTER_FOLDS: List[FoldSpec] = [
    FoldSpec("Fold 1", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-08")),
    FoldSpec("Fold 2", pd.Timestamp("2026-01-08"), pd.Timestamp("2026-01-15")),
    FoldSpec("Fold 3", pd.Timestamp("2026-01-15"), pd.Timestamp("2026-01-21")),
    FoldSpec("Fold 4", pd.Timestamp("2026-01-21"), pd.Timestamp("2026-01-28")),
    FoldSpec("Fold 5", pd.Timestamp("2026-01-28"), pd.Timestamp("2026-02-04")),
    FoldSpec("Fold 6", pd.Timestamp("2026-02-04"), pd.Timestamp("2026-02-11")),
    FoldSpec("Fold 7", pd.Timestamp("2026-02-11"), pd.Timestamp("2026-02-18")),
]


SOURCE_STATS = [
    "adj_o",
    "adj_d",
    "barthag",
    "efg",
    "efgd",
    "tor",
    "tord",
    "orb",
    "drb",
    "adj_tempo",
    "wab",
    "recent_luck_trend",
    "recent_form",
    "two_pt_pct",
    "two_pt_pct_d",
    "three_pt_pct",
    "three_pt_pct_d",
    "three_pt_rate",
    "three_pt_rate_d",
    "ftr",
    "ftrd",
    "days_since_last_game",
    "games_last_7_days",
    "adj_o_ewm5",
    "adj_d_ewm5",
    "barthag_ewm5",
    "efg_ewm5",
    "efgd_ewm5",
    "tor_ewm5",
    "tord_ewm5",
    "adj_tempo_ewm5",
    "delta14_adj_o",
    "delta14_adj_d",
    "delta14_barthag",
    "delta14_efg",
    "delta14_tor",
    "delta14_wab",
    "std14_adj_o",
    "std14_adj_d",
    "std14_efg",
    "std14_tempo",
]


SNAPSHOT_STATS = [
    "adj_o",
    "adj_d",
    "barthag",
    "efg",
    "efgd",
    "tor",
    "tord",
    "orb",
    "drb",
    "adj_tempo",
    "wab",
    "recent_luck_trend",
    "recent_form",
    "two_pt_pct",
    "two_pt_pct_d",
    "three_pt_pct",
    "three_pt_pct_d",
    "three_pt_rate",
    "three_pt_rate_d",
    "ftr",
    "ftrd",
    "adj_o_ewm5",
    "adj_d_ewm5",
    "barthag_ewm5",
    "efg_ewm5",
    "efgd_ewm5",
    "tor_ewm5",
    "tord_ewm5",
    "adj_tempo_ewm5",
    "delta14_adj_o",
    "delta14_adj_d",
    "delta14_barthag",
    "delta14_efg",
    "delta14_tor",
    "delta14_wab",
    "std14_adj_o",
    "std14_adj_d",
    "std14_efg",
    "std14_tempo",
]


def add_luck_regression_signals(torvik: pd.DataFrame) -> pd.DataFrame:
    out = torvik.copy()
    out["date_dt"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date_dt"]).copy()

    required_numeric = [
        "adj_o",
        "adj_d",
        "barthag",
        "efg",
        "efgd",
        "tor",
        "tord",
        "adj_tempo",
        "wab",
    ]
    for col in required_numeric:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["adj_net"] = out["adj_o"].fillna(0.0) - out["adj_d"].fillna(0.0)

    out = out.sort_values(["team_id", "date_dt"]).reset_index(drop=True)
    out["wab_lag7"] = out.groupby("team_id")["wab"].shift(7)
    out["adj_net_lag7"] = out.groupby("team_id")["adj_net"].shift(7)
    out["recent_luck_trend"] = (out["wab"] - out["wab_lag7"]).fillna(0.0)
    out["recent_form"] = (out["adj_net"] - out["adj_net_lag7"]).fillna(0.0)

    # RF1: smooth recent level.
    ewm5_cols = [
        "adj_o",
        "adj_d",
        "barthag",
        "efg",
        "efgd",
        "tor",
        "tord",
        "adj_tempo",
    ]
    for col in ewm5_cols:
        out[f"{col}_ewm5"] = out.groupby("team_id")[col].transform(
            lambda s: s.ewm(span=5, adjust=False, min_periods=1).mean()
        )

    # RF2: momentum over 14 observations.
    delta14_cols = {
        "delta14_adj_o": "adj_o",
        "delta14_adj_d": "adj_d",
        "delta14_barthag": "barthag",
        "delta14_efg": "efg",
        "delta14_tor": "tor",
        "delta14_wab": "wab",
    }
    for new_col, base_col in delta14_cols.items():
        lagged = out.groupby("team_id")[base_col].shift(14)
        out[new_col] = (out[base_col] - lagged).fillna(0.0)

    # RF3: rolling volatility over 14 observations.
    std14_cols = {
        "std14_adj_o": "adj_o",
        "std14_adj_d": "adj_d",
        "std14_efg": "efg",
        "std14_tempo": "adj_tempo",
    }
    for new_col, base_col in std14_cols.items():
        out[new_col] = out.groupby("team_id")[base_col].transform(
            lambda s: s.rolling(window=14, min_periods=5).std()
        ).fillna(0.0)

    out["date"] = out["date_dt"].dt.strftime("%Y-%m-%d")
    return out


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    acc_teams = pd.read_csv(ACC_TEAMS_PATH)
    torvik_path = TORVIK_ALL_PATH if os.path.exists(TORVIK_ALL_PATH) else TORVIK_FALLBACK_PATH
    torvik = pd.read_csv(torvik_path)
    all_games = pd.read_csv(ALL_GAMES_PATH)

    torvik["team_id"] = pd.to_numeric(torvik["team_id"], errors="coerce")
    torvik = torvik[torvik["team_id"].notna()].copy()
    torvik["team_id"] = torvik["team_id"].astype(int)
    torvik = add_luck_regression_signals(torvik)
    return acc_teams, torvik, all_games


def calculate_rest_days(games_df: pd.DataFrame, all_games_df: pd.DataFrame) -> pd.DataFrame:
    out_games = games_df.copy()
    all_games = all_games_df.copy()
    all_games["date_dt"] = pd.to_datetime(all_games["date"], errors="coerce")
    all_games = all_games.dropna(subset=["date_dt"]).copy()

    team_games = []
    for _, row in all_games.iterrows():
        team_games.append({"team_id": int(row["home_team_id"]), "date_dt": row["date_dt"]})
        team_games.append({"team_id": int(row["away_team_id"]), "date_dt": row["date_dt"]})

    tg = pd.DataFrame(team_games).sort_values(["team_id", "date_dt"]).reset_index(drop=True)
    tg["prev_date_dt"] = tg.groupby("team_id")["date_dt"].shift(1)
    tg["days_since_last"] = (tg["date_dt"] - tg["prev_date_dt"]).dt.days

    rest_lookup = tg.set_index(["team_id", "date_dt"])["days_since_last"].to_dict()
    out_games["date_dt"] = pd.to_datetime(out_games["date"], errors="coerce")

    def get_rest(team_id: int, date_dt: pd.Timestamp) -> float:
        key = (int(team_id), date_dt)
        return rest_lookup.get(key, np.nan)

    out_games["home_rest"] = out_games.apply(
        lambda r: get_rest(r["home_team_id"], r["date_dt"]), axis=1
    )
    out_games["away_rest"] = out_games.apply(
        lambda r: get_rest(r["away_team_id"], r["date_dt"]), axis=1
    )

    mean_rest = out_games[["home_rest", "away_rest"]].stack().mean()
    if np.isnan(mean_rest):
        mean_rest = 3.0
    out_games["home_rest"] = out_games["home_rest"].fillna(mean_rest)
    out_games["away_rest"] = out_games["away_rest"].fillna(mean_rest)
    return out_games


def merge_torvik_asof(games: pd.DataFrame, torvik: pd.DataFrame, stat_cols: List[str]) -> pd.DataFrame:
    out = games.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["as_of_date"] = (pd.to_datetime(out["date"]) - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")

    tv = torvik.copy()
    tv["date"] = pd.to_datetime(tv["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    available = [c for c in stat_cols if c in tv.columns]
    home_cols = {c: f"home_{c}" for c in available}
    away_cols = {c: f"away_{c}" for c in available}

    home = tv[["team_id", "date"] + available].rename(columns=home_cols)
    home = home.rename(columns={"team_id": "home_team_id", "date": "as_of_date"})
    away = tv[["team_id", "date"] + available].rename(columns=away_cols)
    away = away.rename(columns={"team_id": "away_team_id", "date": "as_of_date"})

    out = out.merge(home, on=["home_team_id", "as_of_date"], how="left")
    out = out.merge(away, on=["away_team_id", "as_of_date"], how="left")
    return out


def apply_fixed_cutoff_snapshot(
    df: pd.DataFrame, torvik: pd.DataFrame, cutoff_date: pd.Timestamp, snapshot_stats: List[str]
) -> pd.DataFrame:
    out = df.copy()
    out["date_dt"] = pd.to_datetime(out["date"], errors="coerce")
    future_mask = out["date_dt"] > cutoff_date
    if not future_mask.any():
        return out

    tv_cut = torvik.copy()
    tv_cut["date_dt"] = pd.to_datetime(tv_cut["date"], errors="coerce")
    tv_cut = tv_cut[tv_cut["date_dt"] <= cutoff_date]
    tv_cut = tv_cut.sort_values("date_dt").groupby("team_id", as_index=False).tail(1)
    tv_cut_indexed = tv_cut.set_index("team_id")

    for stat in snapshot_stats:
        home_col = f"home_{stat}"
        away_col = f"away_{stat}"
        if stat not in tv_cut_indexed.columns:
            continue
        if home_col not in out.columns or away_col not in out.columns:
            continue
        snapshot = tv_cut_indexed[stat].to_dict()
        out.loc[future_mask, home_col] = out.loc[future_mask, "home_team_id"].map(snapshot)
        out.loc[future_mask, away_col] = out.loc[future_mask, "away_team_id"].map(snapshot)
    return out


def add_recent_opponent_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "event_id" not in out.columns:
        out["event_id"] = np.arange(len(out), dtype=int)
    out["date_dt"] = pd.to_datetime(out["date_dt"], errors="coerce")
    out = out.sort_values(["date_dt", "event_id"]).reset_index(drop=True)

    out["home_barthag"] = pd.to_numeric(out.get("home_barthag"), errors="coerce")
    out["away_barthag"] = pd.to_numeric(out.get("away_barthag"), errors="coerce")
    out["home_adj_o"] = pd.to_numeric(out.get("home_adj_o"), errors="coerce")
    out["away_adj_o"] = pd.to_numeric(out.get("away_adj_o"), errors="coerce")
    out["home_adj_d"] = pd.to_numeric(out.get("home_adj_d"), errors="coerce")
    out["away_adj_d"] = pd.to_numeric(out.get("away_adj_d"), errors="coerce")

    home_apps = out[
        ["event_id", "date_dt", "home_team_id", "away_barthag", "away_adj_o", "away_adj_d"]
    ].rename(
        columns={
            "home_team_id": "team_id",
            "away_barthag": "opp_barthag",
            "away_adj_o": "opp_adj_o",
            "away_adj_d": "opp_adj_d",
        }
    )
    away_apps = out[
        ["event_id", "date_dt", "away_team_id", "home_barthag", "home_adj_o", "home_adj_d"]
    ].rename(
        columns={
            "away_team_id": "team_id",
            "home_barthag": "opp_barthag",
            "home_adj_o": "opp_adj_o",
            "home_adj_d": "opp_adj_d",
        }
    )
    apps = pd.concat([home_apps, away_apps], ignore_index=True)
    apps = apps.sort_values(["team_id", "date_dt", "event_id"]).reset_index(drop=True)

    metric_pairs = [
        ("opp_barthag", "opp_avg_barthag_last5"),
        ("opp_adj_o", "opp_avg_adj_o_last5"),
        ("opp_adj_d", "opp_avg_adj_d_last5"),
    ]
    for source_col, out_col in metric_pairs:
        apps[out_col] = apps.groupby("team_id")[source_col].transform(
            lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
        )

    keep_cols = ["event_id", "team_id"] + [c for _, c in metric_pairs]
    home_merge = apps[keep_cols].rename(
        columns={
            "team_id": "home_team_id",
            "opp_avg_barthag_last5": "home_opp_avg_barthag_last5",
            "opp_avg_adj_o_last5": "home_opp_avg_adj_o_last5",
            "opp_avg_adj_d_last5": "home_opp_avg_adj_d_last5",
        }
    )
    away_merge = apps[keep_cols].rename(
        columns={
            "team_id": "away_team_id",
            "opp_avg_barthag_last5": "away_opp_avg_barthag_last5",
            "opp_avg_adj_o_last5": "away_opp_avg_adj_o_last5",
            "opp_avg_adj_d_last5": "away_opp_avg_adj_d_last5",
        }
    )
    out = out.merge(home_merge, on=["event_id", "home_team_id"], how="left")
    out = out.merge(away_merge, on=["event_id", "away_team_id"], how="left")
    return out


def add_venue_split_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "event_id" not in out.columns:
        out["event_id"] = np.arange(len(out), dtype=int)
    out["date_dt"] = pd.to_datetime(out["date_dt"], errors="coerce")
    out["margin_num"] = pd.to_numeric(out["margin"], errors="coerce")
    out = out.sort_values(["date_dt", "event_id"]).reset_index(drop=True)

    home_hist = out[["event_id", "date_dt", "home_team_id", "margin_num"]].copy()
    home_hist = home_hist.sort_values(["home_team_id", "date_dt", "event_id"]).reset_index(drop=True)
    g_home = home_hist.groupby("home_team_id")
    home_prior_sum = g_home["margin_num"].cumsum() - home_hist["margin_num"]
    home_prior_n = g_home.cumcount()
    home_hist["home_home_net_rating"] = home_prior_sum / home_prior_n.replace(0, np.nan)

    away_hist = out[["event_id", "date_dt", "away_team_id", "margin_num"]].copy()
    away_hist["away_margin_num"] = -away_hist["margin_num"]
    away_hist = away_hist.sort_values(["away_team_id", "date_dt", "event_id"]).reset_index(drop=True)
    g_away = away_hist.groupby("away_team_id")
    away_prior_sum = g_away["away_margin_num"].cumsum() - away_hist["away_margin_num"]
    away_prior_n = g_away.cumcount()
    away_hist["away_away_net_rating"] = away_prior_sum / away_prior_n.replace(0, np.nan)

    out = out.merge(home_hist[["event_id", "home_home_net_rating"]], on="event_id", how="left")
    out = out.merge(away_hist[["event_id", "away_away_net_rating"]], on="event_id", how="left")
    out = out.drop(columns=["margin_num"])
    return out


def lock_non_torvik_features_to_fold_cutoff(df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    out["date_dt"] = pd.to_datetime(out["date_dt"], errors="coerce")
    future_mask = out["date_dt"] > cutoff_date
    if not future_mask.any():
        return out

    hist = out[out["date_dt"] <= cutoff_date].copy()

    def combined_team_snapshot(home_col: str, away_col: str) -> Dict[int, float]:
        home_vals = hist[["home_team_id", "date_dt", home_col]].rename(
            columns={"home_team_id": "team_id", home_col: "value"}
        )
        away_vals = hist[["away_team_id", "date_dt", away_col]].rename(
            columns={"away_team_id": "team_id", away_col: "value"}
        )
        stack = pd.concat([home_vals, away_vals], ignore_index=True)
        stack = stack.dropna(subset=["value"]).sort_values(["team_id", "date_dt"])
        if len(stack) == 0:
            return {}
        return stack.groupby("team_id", as_index=False).tail(1).set_index("team_id")["value"].to_dict()

    for base in ["opp_avg_barthag_last5", "opp_avg_adj_o_last5", "opp_avg_adj_d_last5"]:
        home_col = f"home_{base}"
        away_col = f"away_{base}"
        if home_col not in out.columns or away_col not in out.columns:
            continue
        snap = combined_team_snapshot(home_col, away_col)
        out.loc[future_mask, home_col] = out.loc[future_mask, "home_team_id"].map(snap)
        out.loc[future_mask, away_col] = out.loc[future_mask, "away_team_id"].map(snap)

    if "home_home_net_rating" in out.columns:
        home_snap = (
            hist[["home_team_id", "date_dt", "home_home_net_rating"]]
            .dropna(subset=["home_home_net_rating"])
            .sort_values(["home_team_id", "date_dt"])
        )
        home_map = (
            home_snap.groupby("home_team_id", as_index=False)
            .tail(1)
            .set_index("home_team_id")["home_home_net_rating"]
            .to_dict()
        )
        out.loc[future_mask, "home_home_net_rating"] = out.loc[future_mask, "home_team_id"].map(home_map)

    if "away_away_net_rating" in out.columns:
        away_snap = (
            hist[["away_team_id", "date_dt", "away_away_net_rating"]]
            .dropna(subset=["away_away_net_rating"])
            .sort_values(["away_team_id", "date_dt"])
        )
        away_map = (
            away_snap.groupby("away_team_id", as_index=False)
            .tail(1)
            .set_index("away_team_id")["away_away_net_rating"]
            .to_dict()
        )
        out.loc[future_mask, "away_away_net_rating"] = out.loc[future_mask, "away_team_id"].map(away_map)

    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def col_num(col_name: str) -> pd.Series:
        if col_name in out.columns:
            return pd.to_numeric(out[col_name], errors="coerce")
        return pd.Series(np.nan, index=out.index, dtype=float)

    out["home_adj_o"] = col_num("home_adj_o")
    out["away_adj_o"] = col_num("away_adj_o")
    out["home_adj_d"] = col_num("home_adj_d")
    out["away_adj_d"] = col_num("away_adj_d")

    out["home_efg"] = col_num("home_efg")
    out["away_efg"] = col_num("away_efg")
    out["home_efgd"] = col_num("home_efgd")
    out["away_efgd"] = col_num("away_efgd")

    out["home_tor"] = col_num("home_tor")
    out["away_tor"] = col_num("away_tor")
    out["home_tord"] = col_num("home_tord")
    out["away_tord"] = col_num("away_tord")
    out["home_orb"] = col_num("home_orb")
    out["away_orb"] = col_num("away_orb")
    out["home_drb"] = col_num("home_drb")
    out["away_drb"] = col_num("away_drb")

    out["home_adj_tempo"] = col_num("home_adj_tempo")
    out["away_adj_tempo"] = col_num("away_adj_tempo")
    out["home_barthag"] = col_num("home_barthag")
    out["away_barthag"] = col_num("away_barthag")

    out["home_two_pt_pct"] = col_num("home_two_pt_pct")
    out["away_two_pt_pct"] = col_num("away_two_pt_pct")
    out["home_two_pt_pct_d"] = col_num("home_two_pt_pct_d")
    out["away_two_pt_pct_d"] = col_num("away_two_pt_pct_d")
    out["home_three_pt_pct"] = col_num("home_three_pt_pct")
    out["away_three_pt_pct"] = col_num("away_three_pt_pct")
    out["home_three_pt_pct_d"] = col_num("home_three_pt_pct_d")
    out["away_three_pt_pct_d"] = col_num("away_three_pt_pct_d")
    out["home_three_pt_rate"] = col_num("home_three_pt_rate")
    out["away_three_pt_rate"] = col_num("away_three_pt_rate")
    out["home_three_pt_rate_d"] = col_num("home_three_pt_rate_d")
    out["away_three_pt_rate_d"] = col_num("away_three_pt_rate_d")

    out["home_ftr"] = col_num("home_ftr")
    out["away_ftr"] = col_num("away_ftr")
    out["home_ftrd"] = col_num("home_ftrd")
    out["away_ftrd"] = col_num("away_ftrd")

    out["home_wab"] = col_num("home_wab").fillna(0.0)
    out["away_wab"] = col_num("away_wab").fillna(0.0)
    out["home_recent_luck_trend"] = col_num("home_recent_luck_trend").fillna(0.0)
    out["away_recent_luck_trend"] = col_num("away_recent_luck_trend").fillna(0.0)
    out["home_recent_form"] = col_num("home_recent_form").fillna(0.0)
    out["away_recent_form"] = col_num("away_recent_form").fillna(0.0)

    rolling_cols = [
        "adj_o_ewm5",
        "adj_d_ewm5",
        "barthag_ewm5",
        "efg_ewm5",
        "efgd_ewm5",
        "tor_ewm5",
        "tord_ewm5",
        "adj_tempo_ewm5",
        "delta14_adj_o",
        "delta14_adj_d",
        "delta14_barthag",
        "delta14_efg",
        "delta14_tor",
        "delta14_wab",
        "std14_adj_o",
        "std14_adj_d",
        "std14_efg",
        "std14_tempo",
    ]
    for col in rolling_cols:
        out[f"home_{col}"] = col_num(f"home_{col}")
        out[f"away_{col}"] = col_num(f"away_{col}")

    out["home_opp_avg_barthag_last5"] = col_num("home_opp_avg_barthag_last5")
    out["away_opp_avg_barthag_last5"] = col_num("away_opp_avg_barthag_last5")
    out["home_opp_avg_adj_o_last5"] = col_num("home_opp_avg_adj_o_last5")
    out["away_opp_avg_adj_o_last5"] = col_num("away_opp_avg_adj_o_last5")
    out["home_opp_avg_adj_d_last5"] = col_num("home_opp_avg_adj_d_last5")
    out["away_opp_avg_adj_d_last5"] = col_num("away_opp_avg_adj_d_last5")
    out["home_home_net_rating"] = col_num("home_home_net_rating")
    out["away_away_net_rating"] = col_num("away_away_net_rating")

    out["adj_o_diff"] = out["home_adj_o"] - out["away_adj_o"]
    out["adj_d_diff"] = out["home_adj_d"] - out["away_adj_d"]
    out["efg_diff"] = out["home_efg"] - out["away_efg"]
    out["efgd_diff"] = out["home_efgd"] - out["away_efgd"]
    out["is_neutral"] = pd.to_numeric(out["neutral_site"], errors="coerce").fillna(0).astype(int)

    out["tor_diff"] = out["home_tor"] - out["away_tor"]
    out["tord_diff"] = out["home_tord"] - out["away_tord"]
    out["orb_diff"] = out["home_orb"] - out["away_orb"]
    out["drb_diff"] = out["home_drb"] - out["away_drb"]

    out["adj_tempo_diff"] = out["home_adj_tempo"] - out["away_adj_tempo"]
    out["adj_tempo_avg"] = (out["home_adj_tempo"] + out["away_adj_tempo"]) / 2.0

    out["tempo_x_tor_diff"] = out["adj_tempo_avg"] * out["tor_diff"]
    out["tempo_x_tord_diff"] = out["adj_tempo_avg"] * out["tord_diff"]
    out["tempo_x_orb_diff"] = out["adj_tempo_avg"] * out["orb_diff"]
    out["tempo_x_drb_diff"] = out["adj_tempo_avg"] * out["drb_diff"]

    out["two_pt_pct_diff"] = out["home_two_pt_pct"] - out["away_two_pt_pct"]
    out["two_pt_pct_d_diff"] = out["home_two_pt_pct_d"] - out["away_two_pt_pct_d"]
    out["three_pt_pct_diff"] = out["home_three_pt_pct"] - out["away_three_pt_pct"]
    out["three_pt_pct_d_diff"] = out["home_three_pt_pct_d"] - out["away_three_pt_pct_d"]
    out["three_pt_rate_diff"] = out["home_three_pt_rate"] - out["away_three_pt_rate"]
    out["three_pt_rate_d_diff"] = out["home_three_pt_rate_d"] - out["away_three_pt_rate_d"]

    out["home_3p_mismatch"] = out["home_three_pt_pct"] - out["away_three_pt_pct_d"]
    out["away_3p_mismatch"] = out["away_three_pt_pct"] - out["home_three_pt_pct_d"]
    out["home_2p_mismatch"] = out["home_two_pt_pct"] - out["away_two_pt_pct_d"]
    out["away_2p_mismatch"] = out["away_two_pt_pct"] - out["home_two_pt_pct_d"]

    out["three_pt_style_mismatch_diff"] = out["home_3p_mismatch"] - out["away_3p_mismatch"]
    out["two_pt_style_mismatch_diff"] = out["home_2p_mismatch"] - out["away_2p_mismatch"]
    out["shot_profile_asymmetry"] = (
        out["three_pt_style_mismatch_diff"] - out["two_pt_style_mismatch_diff"]
    )
    out["rate_x_pct_mismatch"] = out["three_pt_rate_diff"] * out["three_pt_style_mismatch_diff"]

    out["tempo_x_adj_o_diff"] = out["adj_tempo_avg"] * out["adj_o_diff"]
    out["tempo_x_adj_d_diff"] = out["adj_tempo_avg"] * out["adj_d_diff"]
    out["tempo_x_efg_diff"] = out["adj_tempo_avg"] * out["efg_diff"]
    out["tempo_x_efgd_diff"] = out["adj_tempo_avg"] * out["efgd_diff"]

    out["adj_o_ewm5_diff"] = out["home_adj_o_ewm5"] - out["away_adj_o_ewm5"]
    out["adj_d_ewm5_diff"] = out["home_adj_d_ewm5"] - out["away_adj_d_ewm5"]
    out["barthag_ewm5_diff"] = out["home_barthag_ewm5"] - out["away_barthag_ewm5"]
    out["efg_ewm5_diff"] = out["home_efg_ewm5"] - out["away_efg_ewm5"]
    out["efgd_ewm5_diff"] = out["home_efgd_ewm5"] - out["away_efgd_ewm5"]
    out["tor_ewm5_diff"] = out["home_tor_ewm5"] - out["away_tor_ewm5"]
    out["tord_ewm5_diff"] = out["home_tord_ewm5"] - out["away_tord_ewm5"]
    out["adj_tempo_ewm5_diff"] = out["home_adj_tempo_ewm5"] - out["away_adj_tempo_ewm5"]

    out["delta14_adj_o_diff"] = out["home_delta14_adj_o"] - out["away_delta14_adj_o"]
    out["delta14_adj_d_diff"] = out["home_delta14_adj_d"] - out["away_delta14_adj_d"]
    out["delta14_barthag_diff"] = out["home_delta14_barthag"] - out["away_delta14_barthag"]
    out["delta14_efg_diff"] = out["home_delta14_efg"] - out["away_delta14_efg"]
    out["delta14_tor_diff"] = out["home_delta14_tor"] - out["away_delta14_tor"]
    out["delta14_wab_diff"] = out["home_delta14_wab"] - out["away_delta14_wab"]

    out["std14_adj_o_avg"] = (out["home_std14_adj_o"] + out["away_std14_adj_o"]) / 2.0
    out["std14_adj_d_avg"] = (out["home_std14_adj_d"] + out["away_std14_adj_d"]) / 2.0
    out["std14_efg_avg"] = (out["home_std14_efg"] + out["away_std14_efg"]) / 2.0
    out["std14_tempo_avg"] = (out["home_std14_tempo"] + out["away_std14_tempo"]) / 2.0

    out["opp_avg_barthag_last5_diff"] = (
        out["home_opp_avg_barthag_last5"] - out["away_opp_avg_barthag_last5"]
    )
    out["opp_avg_adj_o_last5_diff"] = (
        out["home_opp_avg_adj_o_last5"] - out["away_opp_avg_adj_o_last5"]
    )
    out["opp_avg_adj_d_last5_diff"] = (
        out["home_opp_avg_adj_d_last5"] - out["away_opp_avg_adj_d_last5"]
    )
    out["venue_split_edge"] = out["home_home_net_rating"] - out["away_away_net_rating"]

    out["home_tor_mismatch"] = out["home_tor"] - out["away_tord"]
    out["away_tor_mismatch"] = out["away_tor"] - out["home_tord"]
    out["tor_mismatch"] = out["home_tor_mismatch"] - out["away_tor_mismatch"]
    out["home_orb_mismatch"] = out["home_orb"] - out["away_drb"]
    out["away_orb_mismatch"] = out["away_orb"] - out["home_drb"]
    out["orb_mismatch"] = out["home_orb_mismatch"] - out["away_orb_mismatch"]

    out["home_rest"] = pd.to_numeric(out["home_rest"], errors="coerce")
    out["away_rest"] = pd.to_numeric(out["away_rest"], errors="coerce")
    out["rest_diff"] = out["home_rest"] - out["away_rest"]
    out["home_is_b2b"] = (out["home_rest"] <= 1.0).astype(float)
    out["away_is_b2b"] = (out["away_rest"] <= 1.0).astype(float)
    out["b2b_diff"] = out["home_is_b2b"] - out["away_is_b2b"]
    out["home_short_rest"] = (out["home_rest"] <= 2.0).astype(float)
    out["away_short_rest"] = (out["away_rest"] <= 2.0).astype(float)
    out["short_rest_diff"] = out["home_short_rest"] - out["away_short_rest"]
    out["home_long_rest"] = (out["home_rest"] >= 5.0).astype(float)
    out["away_long_rest"] = (out["away_rest"] >= 5.0).astype(float)
    out["long_rest_diff"] = out["home_long_rest"] - out["away_long_rest"]

    out["home_days_since_last_game"] = col_num("home_days_since_last_game").fillna(out["home_rest"])
    out["away_days_since_last_game"] = col_num("away_days_since_last_game").fillna(out["away_rest"])
    out["home_games_last_7_days"] = col_num("home_games_last_7_days").fillna(1.0)
    out["away_games_last_7_days"] = col_num("away_games_last_7_days").fillna(1.0)

    out["days_since_last_game_diff"] = (
        out["home_days_since_last_game"] - out["away_days_since_last_game"]
    )
    out["games_last_7_days_diff"] = (
        out["home_games_last_7_days"] - out["away_games_last_7_days"]
    )
    out["days_since_last_game_avg"] = (
        out["home_days_since_last_game"] + out["away_days_since_last_game"]
    ) / 2.0
    out["games_last_7_days_avg"] = (
        out["home_games_last_7_days"] + out["away_games_last_7_days"]
    ) / 2.0

    out["home_fatigue_index"] = (
        out["home_games_last_7_days"] / np.maximum(out["home_days_since_last_game"], 1.0)
    )
    out["away_fatigue_index"] = (
        out["away_games_last_7_days"] / np.maximum(out["away_days_since_last_game"], 1.0)
    )
    out["fatigue_index_avg"] = (out["home_fatigue_index"] + out["away_fatigue_index"]) / 2.0
    out["fatigue_index_diff"] = out["home_fatigue_index"] - out["away_fatigue_index"]

    out["three_pt_rate_avg"] = (out["home_three_pt_rate"] + out["away_three_pt_rate"]) / 2.0
    out["three_pt_pct_avg"] = (out["home_three_pt_pct"] + out["away_three_pt_pct"]) / 2.0
    out["var_proxy_tempo_3pr"] = out["three_pt_rate_avg"] * out["adj_tempo_avg"]
    out["var_proxy_3pr_1m3pp"] = out["three_pt_rate_avg"] * (1.0 - out["three_pt_pct_avg"] / 100.0)
    out["var_proxy_tempo_3pp_gap"] = out["adj_tempo_avg"] * out["three_pt_pct_diff"].abs()

    out["ftr_diff"] = out["home_ftr"] - out["away_ftr"]
    out["ftr_allowed_diff"] = out["home_ftrd"] - out["away_ftrd"]
    out["home_foul_pressure"] = out["home_ftr"] - out["away_ftrd"]
    out["away_foul_pressure"] = out["away_ftr"] - out["home_ftrd"]
    out["foul_rate_diff"] = out["home_foul_pressure"] - out["away_foul_pressure"]
    out["opponent_foul_rate_diff"] = out["away_ftrd"] - out["home_ftrd"]

    out["close_spread_proxy"] = 1.0 / (1.0 + (out["adj_o_diff"] - out["adj_d_diff"]).abs())
    out["ftr_x_close_spread"] = out["ftr_diff"] * out["close_spread_proxy"]
    out["ftr_x_fatigue"] = out["ftr_diff"] * out["fatigue_index_diff"]

    out["luck_diff"] = out["home_wab"] - out["away_wab"]
    out["recent_luck_trend_diff"] = out["home_recent_luck_trend"] - out["away_recent_luck_trend"]
    out["recent_form_diff"] = out["home_recent_form"] - out["away_recent_form"]
    out["luck_x_recent_form"] = out["luck_diff"] * out["recent_form_diff"]
    out["luck_reversion_term"] = -out["luck_diff"]
    out["luck_x_tempo"] = out["luck_diff"] * out["adj_tempo_avg"]

    return out


def choose_tscv_splits(n_rows: int) -> int:
    if n_rows >= 1000:
        return 5
    if n_rows >= 400:
        return 4
    if n_rows >= 200:
        return 3
    return 2


def fit_ridge_for_fold(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    train_mask: pd.Series,
    test_mask: pd.Series,
    target: str = "margin",
) -> Dict[str, object]:
    work = df.copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=[target]).copy()

    train_mask_work = train_mask.reindex(work.index).fillna(False).astype(bool)
    test_mask_work = test_mask.reindex(work.index).fillna(False).astype(bool)

    train_n = int(train_mask_work.sum())
    test_n = int(test_mask_work.sum())
    if train_n < MIN_TRAIN_ROWS:
        raise ValueError(f"Insufficient train rows ({train_n}) for ridge fit.")
    if test_n < MIN_TEST_ROWS:
        raise ValueError(f"Insufficient test rows ({test_n}) for ridge fit.")

    X_train = work.loc[train_mask_work, list(feature_cols)]
    X_test = work.loc[test_mask_work, list(feature_cols)]
    y_train = work.loc[train_mask_work, target].to_numpy(dtype=float)
    y_test = work.loc[test_mask_work, target].to_numpy(dtype=float)

    n_splits = min(choose_tscv_splits(train_n), train_n - 1)
    if n_splits < 2:
        raise ValueError("Need at least 3 train rows for inner time-series CV.")

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ]
    )
    inner_cv = TimeSeriesSplit(n_splits=n_splits)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"ridge__alpha": ALPHA_GRID},
        cv=inner_cv,
        scoring="neg_mean_absolute_error",
        refit=True,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_abs_error = np.abs(y_train - train_pred)
    test_abs_error = np.abs(y_test - test_pred)

    return {
        "model": model,
        "alpha": float(search.best_params_["ridge__alpha"]),
        "feature_cols": list(feature_cols),
        "train_index": work.loc[train_mask_work].index.to_numpy(),
        "test_index": work.loc[test_mask_work].index.to_numpy(),
        "y_train": y_train,
        "y_test": y_test,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "train_abs_error": train_abs_error,
        "test_abs_error": test_abs_error,
        "train_mae": float(mean_absolute_error(y_train, train_pred)),
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
    }


def build_base_model_frame() -> Tuple[pd.DataFrame, pd.DataFrame]:
    acc_teams, torvik, all_games = load_data()
    _ = acc_teams

    games = all_games.copy()
    games["home_team_id"] = pd.to_numeric(games["home_team_id"], errors="coerce")
    games["away_team_id"] = pd.to_numeric(games["away_team_id"], errors="coerce")
    games = games.dropna(subset=["home_team_id", "away_team_id"]).copy()
    games["home_team_id"] = games["home_team_id"].astype(int)
    games["away_team_id"] = games["away_team_id"].astype(int)

    torvik_ids = set(torvik["team_id"].astype(int).tolist())
    mapped_mask = games["home_team_id"].isin(torvik_ids) & games["away_team_id"].isin(torvik_ids)
    excluded = int((~mapped_mask).sum())
    games = games[mapped_mask].reset_index(drop=True)

    games = calculate_rest_days(games, all_games)
    merged = merge_torvik_asof(games, torvik, SOURCE_STATS)
    merged["date_dt"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.dropna(subset=["date_dt", "margin"]).sort_values("date_dt").reset_index(drop=True)
    merged = add_recent_opponent_quality_features(merged)
    merged = add_venue_split_features(merged)

    print(
        f"Loaded {len(all_games)} total games, {len(merged)} Torvik-covered model rows "
        f"(excluded {excluded} unmapped rows)."
    )
    return merged, torvik


def build_cumulative_feature_blocks(
    blocks: Sequence[Tuple[str, Sequence[str]]],
) -> List[Tuple[str, List[str]]]:
    cumulative: List[str] = []
    seen = set()
    out: List[Tuple[str, List[str]]] = []

    for block_name, block_features in blocks:
        for feat in block_features:
            if feat not in seen:
                cumulative.append(feat)
                seen.add(feat)
        out.append((block_name, list(cumulative)))
    return out


def one_sided_paired_permutation_pvalue(
    prev_abs_error: np.ndarray,
    curr_abs_error: np.ndarray,
    n_permutations: int = P_VALUE_PERMUTATIONS,
    seed: int = RANDOM_SEED,
) -> float:
    prev = np.asarray(prev_abs_error, dtype=float)
    curr = np.asarray(curr_abs_error, dtype=float)
    keep = np.isfinite(prev) & np.isfinite(curr)
    prev = prev[keep]
    curr = curr[keep]
    if len(prev) == 0:
        return np.nan

    diff = prev - curr
    observed = float(np.mean(diff))
    if np.allclose(diff, 0.0):
        return 1.0

    rng = np.random.default_rng(seed)
    ge_count = 0
    for _ in range(n_permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(diff))
        perm_stat = float(np.mean(diff * signs))
        if perm_stat >= observed - 1e-12:
            ge_count += 1
    return float((ge_count + 1) / (n_permutations + 1))


def evaluate_vegas_mae(df: pd.DataFrame, test_mask: pd.Series) -> Tuple[float, int]:
    test_df = df.loc[test_mask].copy()
    if "spread" not in test_df.columns:
        return np.nan, 0
    spread = pd.to_numeric(test_df["spread"], errors="coerce")
    keep = spread.notna()
    if keep.sum() == 0:
        return np.nan, 0
    y_true = pd.to_numeric(test_df.loc[keep, "margin"], errors="coerce").values
    vegas_pred = -spread.loc[keep].values
    mae = float(mean_absolute_error(y_true, vegas_pred))
    return mae, int(keep.sum())


def run_walkforward_block_evaluation(
    base_df: pd.DataFrame,
    torvik: pd.DataFrame,
    folds: Sequence[FoldSpec],
    feature_blocks: Sequence[Tuple[str, Sequence[str]]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cumulative_blocks = build_cumulative_feature_blocks(feature_blocks)
    if len(cumulative_blocks) == 0:
        raise ValueError("feature_blocks must contain at least one block.")

    results_rows: List[Dict[str, object]] = []
    pooled_errors: Dict[str, List[np.ndarray]] = {name: [] for name, _ in cumulative_blocks}
    pooled_base_errors: List[np.ndarray] = []

    for fold_idx, fold in enumerate(folds):
        fold_df = apply_fixed_cutoff_snapshot(base_df, torvik, fold.snapshot_cutoff, SNAPSHOT_STATS)
        fold_df = lock_non_torvik_features_to_fold_cutoff(fold_df, fold.snapshot_cutoff)
        fold_df = add_engineered_features(fold_df).sort_values("date_dt").reset_index(drop=True)

        train_mask = fold_df["date_dt"] < fold.train_end
        test_mask = (fold_df["date_dt"] >= fold.test_start) & (fold_df["date_dt"] < fold.test_end)
        train_n = int(train_mask.sum())
        test_n = int(test_mask.sum())
        if test_n == 0:
            print(f"\n=== {fold.name} skipped: no test rows in [{fold.test_start.date()}, {fold.test_end.date()}) ===")
            continue

        vegas_mae, vegas_n = evaluate_vegas_mae(fold_df, test_mask)
        print(
            f"\n=== {fold.name} | train < {fold.train_end.date()} | "
            f"test [{fold.test_start.date()}, {fold.test_end.date()}) ==="
        )
        print(f"snapshot_cutoff_date_for_test_features: {fold.snapshot_cutoff.date()}")
        print(f"train_n_raw: {train_n}, test_n_raw: {test_n}")

        baseline_result = fit_ridge_for_fold(fold_df, BASE_FEATURES, train_mask, test_mask)
        pooled_base_errors.append(np.asarray(baseline_result["test_abs_error"], dtype=float))
        print(
            f"baseline_3feat_test_mae: {baseline_result['test_mae']:.4f} "
            f"(features={','.join(BASE_FEATURES)})"
        )

        for block_idx, (block_name, feature_cols) in enumerate(cumulative_blocks):
            missing = [c for c in feature_cols if c not in fold_df.columns]
            if missing:
                raise ValueError(f"Missing features for block '{block_name}': {missing}")

            result = fit_ridge_for_fold(fold_df, feature_cols, train_mask, test_mask)
            pooled_errors[block_name].append(np.asarray(result["test_abs_error"], dtype=float))

            mae_delta_vs_base = float(baseline_result["test_mae"] - result["test_mae"])
            p_value_vs_base = one_sided_paired_permutation_pvalue(
                np.asarray(baseline_result["test_abs_error"], dtype=float),
                np.asarray(result["test_abs_error"], dtype=float),
                seed=RANDOM_SEED + fold_idx * 100 + block_idx,
            )

            row = {
                "fold": fold.name,
                "train_end_exclusive": fold.train_end.date().isoformat(),
                "test_start_inclusive": fold.test_start.date().isoformat(),
                "test_end_exclusive": fold.test_end.date().isoformat(),
                "snapshot_cutoff_date": fold.snapshot_cutoff.date().isoformat(),
                "block_name": block_name,
                "n_features": len(feature_cols),
                "train_n": int(len(result["y_train"])),
                "test_n": int(len(result["y_test"])),
                "baseline_3feat_test_mae": float(baseline_result["test_mae"]),
                "best_alpha": float(result["alpha"]),
                "train_mae": float(result["train_mae"]),
                "test_mae": float(result["test_mae"]),
                "mae_delta_vs_base_model": mae_delta_vs_base,
                "p_value_improves_vs_base_model": p_value_vs_base,
                "vegas_test_mae": float(vegas_mae) if np.isfinite(vegas_mae) else np.nan,
                "vegas_n": int(vegas_n),
                "mae_delta_model_minus_vegas": (
                    float(result["test_mae"] - vegas_mae) if np.isfinite(vegas_mae) else np.nan
                ),
            }
            results_rows.append(row)

            delta_txt = "n/a" if np.isnan(mae_delta_vs_base) else f"{mae_delta_vs_base:+.4f}"
            p_txt = "n/a" if np.isnan(p_value_vs_base) else f"{p_value_vs_base:.4f}"
            print(
                f"block={block_name:>20} | features={len(feature_cols):>2d} | "
                f"alpha={result['alpha']:.6f} | test_mae={result['test_mae']:.4f} | "
                f"delta_vs_3feat={delta_txt} | p_value={p_txt}"
            )

    if len(results_rows) == 0:
        raise RuntimeError("No fold results were produced; verify date coverage in all_games.csv.")

    results_df = pd.DataFrame(results_rows)
    pooled_base = (
        np.concatenate(pooled_base_errors) if len(pooled_base_errors) > 0 else np.array([], dtype=float)
    )

    summary_rows: List[Dict[str, object]] = []
    for _, (block_name, feature_cols) in enumerate(cumulative_blocks):
        block_rows = results_df[results_df["block_name"] == block_name]
        pooled_curr = (
            np.concatenate(pooled_errors[block_name])
            if len(pooled_errors[block_name]) > 0
            else np.array([], dtype=float)
        )
        pooled_n = int(len(pooled_curr))

        row = {
            "block_name": block_name,
            "n_features": len(feature_cols),
            "n_folds": int(block_rows["fold"].nunique()),
            "mean_baseline_3feat_test_mae": float(block_rows["baseline_3feat_test_mae"].mean()),
            "mean_test_mae": float(block_rows["test_mae"].mean()),
            "mean_vegas_delta": float(block_rows["mae_delta_model_minus_vegas"].mean())
            if block_rows["mae_delta_model_minus_vegas"].notna().any()
            else np.nan,
            "pooled_test_rows": pooled_n,
            "pooled_mae": float(pooled_curr.mean()) if pooled_n > 0 else np.nan,
            "mean_mae_delta_vs_base_model": float(-block_rows["mae_delta_vs_base_model"].mean()),
            "pooled_mae_delta_vs_base_model": np.nan,
            "pooled_p_value_improves_vs_base_model": np.nan,
        }

        if len(pooled_base) == len(pooled_curr) and len(pooled_curr) > 0:
            row["pooled_mae_delta_vs_base_model"] = float(np.mean(pooled_curr - pooled_base))
            row["pooled_p_value_improves_vs_base_model"] = one_sided_paired_permutation_pvalue(
                pooled_base,
                pooled_curr,
                seed=RANDOM_SEED + 2000,
            )

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    return results_df, summary_df


def main() -> None:
    base_df, torvik = build_base_model_frame()
    print("\nUsing fixed outer folds:")
    for fold in OUTER_FOLDS:
        print(
            f"{fold.name}: train < {fold.train_end.date()} | "
            f"test [{fold.test_start.date()}, {fold.test_end.date()})"
        )

    print("\nFeature blocks (edit FEATURE_BLOCKS to your preferred definitions):")
    for block_name, feats in FEATURE_BLOCKS:
        print(f"{block_name}: {', '.join(feats)}")

    fold_results, block_summary = run_walkforward_block_evaluation(
        base_df=base_df,
        torvik=torvik,
        folds=OUTER_FOLDS,
        feature_blocks=FEATURE_BLOCKS,
    )

    fold_results_path = os.path.join("data", "ridge_walkforward_fold_block_results.csv")
    block_summary_path = os.path.join("data", "ridge_walkforward_block_summary.csv")
    fold_results.to_csv(fold_results_path, index=False)
    block_summary.to_csv(block_summary_path, index=False)

    print("\n=== Block Summary Across Folds ===")
    for _, row in block_summary.iterrows():
        delta_txt = (
            "n/a"
            if pd.isna(row["pooled_mae_delta_vs_base_model"])
            else f"{row['pooled_mae_delta_vs_base_model']:+.4f}"
        )
        p_txt = (
            "n/a"
            if pd.isna(row["pooled_p_value_improves_vs_base_model"])
            else f"{row['pooled_p_value_improves_vs_base_model']:.4f}"
        )
        print(
            f"block={row['block_name']:>20} | mean_test_mae={row['mean_test_mae']:.4f} | "
            f"delta_vs_3feat={delta_txt} | pooled_p_value={p_txt}"
        )

    print(f"\nSaved fold/block results to: {fold_results_path}")
    print(f"Saved block summary to: {block_summary_path}")


if __name__ == "__main__":
    main()
