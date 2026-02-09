import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import model


def add_style_metrics(test_df: pd.DataFrame, torvik: pd.DataFrame) -> pd.DataFrame:
    torvik = torvik.copy()
    torvik["as_of_date"] = pd.to_datetime(torvik["date"]).dt.strftime("%Y-%m-%d")

    home = torvik[["team_id", "as_of_date", "adj_tempo", "wab"]].rename(
        columns={
            "team_id": "home_team_id",
            "adj_tempo": "home_adj_tempo_plot",
            "wab": "home_wab_plot",
        }
    )
    away = torvik[["team_id", "as_of_date", "adj_tempo", "wab"]].rename(
        columns={
            "team_id": "away_team_id",
            "adj_tempo": "away_adj_tempo_plot",
            "wab": "away_wab_plot",
        }
    )

    out = test_df.merge(home, on=["home_team_id", "as_of_date"], how="left")
    out = out.merge(away, on=["away_team_id", "as_of_date"], how="left")

    out["avg_tempo_game"] = (out["home_adj_tempo_plot"] + out["away_adj_tempo_plot"]) / 2.0
    out["wab_diff"] = out["home_wab_plot"] - out["away_wab_plot"]
    out["spread_size"] = out["spread"].abs()
    return out


def scatter_with_trend(ax, x, y, xlabel, title):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    ax.scatter(x, y, alpha=0.35, s=14)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")

    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(np.min(x), np.max(x), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="red", linewidth=2.0)
        corr = np.corrcoef(x, y)[0, 1]
        ax.text(
            0.02,
            0.98,
            f"n={len(x)}\nr={corr:.3f}\nslope={slope:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Residual (Pred - Actual Margin)")
    ax.set_title(title)


def main():
    acc_teams, torvik, all_games = model.load_data()
    acc_team_ids = set(acc_teams["team_id"].astype(int).tolist())
    torvik_team_ids = set(torvik["team_id"].astype(int).tolist())

    games = all_games.copy()
    games["home_team_id"] = games["home_team_id"].astype(int)
    games["away_team_id"] = games["away_team_id"].astype(int)
    games = games[
        games["home_team_id"].isin(torvik_team_ids)
        & games["away_team_id"].isin(torvik_team_ids)
    ].reset_index(drop=True)

    df, feature_cols = model.prepare_features(games, torvik, all_games, acc_team_ids)

    pre_df = df.dropna(subset=["margin"] + feature_cols).copy()
    pre_df["date_dt"] = pd.to_datetime(pre_df["date"])
    pre_df = pre_df.sort_values("date_dt").reset_index(drop=True)
    split_idx = int(len(pre_df) * 0.8)
    cutoff_date = pre_df.loc[split_idx - 1, "date_dt"]

    ridge, scaler, model_df = model.build_model(df, feature_cols, torvik, use_fixed_cutoff=True)
    test_mask = model_df["date_dt"] > cutoff_date
    test_df = model_df.loc[test_mask].copy().reset_index(drop=True)

    X_test = test_df[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    test_df["pred"] = ridge.predict(X_test_scaled)
    test_df["residual"] = test_df["pred"] - test_df["margin"]

    test_df = add_style_metrics(test_df, torvik)

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "ridge_residual_diagnostics.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    spread_mask = test_df["spread_size"].notna()
    scatter_with_trend(
        axes[0],
        test_df.loc[spread_mask, "spread_size"].to_numpy(),
        test_df.loc[spread_mask, "residual"].to_numpy(),
        "Spread Size (|spread|)",
        "Residual vs Spread Size",
    )

    scatter_with_trend(
        axes[1],
        test_df["avg_tempo_game"].to_numpy(),
        test_df["residual"].to_numpy(),
        "Average Team Adj Tempo",
        "Residual vs Tempo",
    )

    scatter_with_trend(
        axes[2],
        test_df["wab_diff"].to_numpy(),
        test_df["residual"].to_numpy(),
        "WAB Differential (Home - Away)",
        "Residual vs WAB Strength Edge",
    )

    fig.suptitle("Ridge Residual Diagnostics (Fixed Cutoff: Base4 + Block2 + Block7)")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"Saved plot: {out_path}")
    print(f"Cutoff date: {cutoff_date.date()}")
    print(f"Test rows: {len(test_df)}")
    print(f"Rows with spread: {int(spread_mask.sum())}")


if __name__ == "__main__":
    main()
