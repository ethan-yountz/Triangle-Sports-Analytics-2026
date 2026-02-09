import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data():
    """Load game and rating data."""
    acc_teams = pd.read_csv(os.path.join("data", "acc_teams.csv"))
    torvik_path = os.path.join("data", "torvik_asof_ratings_all_teams.csv")
    if not os.path.exists(torvik_path):
        torvik_path = os.path.join("data", "torvik_asof_ratings.csv")
    torvik = pd.read_csv(torvik_path)
    torvik["team_id"] = pd.to_numeric(torvik["team_id"], errors="coerce")
    torvik = torvik[torvik["team_id"].notna()].copy()
    torvik["team_id"] = torvik["team_id"].astype(int)
    all_games = pd.read_csv(os.path.join("data", "all_games.csv"))
    return acc_teams, torvik, all_games


def load_team_ids(csv_path, id_col="team_id"):
    """Load team_id values as ints from a CSV."""
    df = pd.read_csv(csv_path)
    if id_col not in df.columns:
        raise ValueError(f"Missing required column '{id_col}' in {csv_path}")
    return set(df[id_col].astype(int).tolist())


def calculate_rest_days(games_df, all_games_df):
    all_games_df = all_games_df.copy()
    all_games_df['date'] = pd.to_datetime(all_games_df['date'])

    team_games = []
    for _, row in all_games_df.iterrows():
        team_games.append({'team_id': row['home_team_id'], 'date': row['date']})
        team_games.append({'team_id': row['away_team_id'], 'date': row['date']})
    
    team_games_df = pd.DataFrame(team_games)
    team_games_df = team_games_df.sort_values(['team_id', 'date'])

    team_games_df['prev_date'] = team_games_df.groupby('team_id')['date'].shift(1)
    team_games_df['days_since_last'] = (team_games_df['date'] - team_games_df['prev_date']).dt.days
    
    rest_lookup = team_games_df.set_index(['team_id', 'date'])['days_since_last'].to_dict()
    
    games_df = games_df.copy()
    games_df['date_dt'] = pd.to_datetime(games_df['date'])
    
    def get_rest(row, team_col):
        team_id = row[team_col]
        date = row['date_dt']
        key = (team_id, date)
        if key in rest_lookup:
            return rest_lookup[key]
        for tid in [team_id, str(team_id), int(float(team_id)) if pd.notna(team_id) else None]:
            if tid is not None:
                key = (tid, date)
                if key in rest_lookup:
                    return rest_lookup[key]
        return None
    
    games_df['home_rest'] = games_df.apply(lambda r: get_rest(r, 'home_team_id'), axis=1)
    games_df['away_rest'] = games_df.apply(lambda r: get_rest(r, 'away_team_id'), axis=1)
    games_df = games_df.drop(columns=['date_dt'])
    
    return games_df


def prepare_features(games, torvik, all_games, acc_team_ids):
    """Merge as-of Torvik stats and engineer model features."""
    games = calculate_rest_days(games, all_games)

    games = games.copy()
    torvik = torvik.copy()
    games['date'] = pd.to_datetime(games['date']).dt.strftime('%Y-%m-%d')
    torvik['date'] = pd.to_datetime(torvik['date']).dt.strftime('%Y-%m-%d')
    games['as_of_date'] = (
        pd.to_datetime(games['date']) - pd.Timedelta(days=1)
    ).dt.strftime('%Y-%m-%d')

    stat_cols = [
        'adj_o',
        'adj_d',
        'efg',
        'tor',
        'tord',
        'orb',
        'drb',
        'days_since_last_game',
        'games_last_7_days',
    ]

    available_stats = [c for c in stat_cols if c in torvik.columns]

    home_cols = {c: f'home_{c}' for c in available_stats}
    home_torvik = torvik[['team_id', 'date'] + available_stats].copy()
    home_torvik = home_torvik.rename(columns=home_cols)
    home_torvik = home_torvik.rename(columns={'team_id': 'home_team_id'})
    home_torvik = home_torvik.rename(columns={'date': 'as_of_date'})

    away_cols = {c: f'away_{c}' for c in available_stats}
    away_torvik = torvik[['team_id', 'date'] + available_stats].copy()
    away_torvik = away_torvik.rename(columns=away_cols)
    away_torvik = away_torvik.rename(columns={'team_id': 'away_team_id'})
    away_torvik = away_torvik.rename(columns={'date': 'as_of_date'})

    df = games.merge(home_torvik, on=['home_team_id', 'as_of_date'], how='left')
    df = df.merge(away_torvik, on=['away_team_id', 'as_of_date'], how='left')

    df['adj_o_diff'] = df['home_adj_o'] - df['away_adj_o']
    df['adj_d_diff'] = df['home_adj_d'] - df['away_adj_d']
    df['efg_diff'] = df['home_efg'] - df['away_efg']

    df['tor_diff'] = df['home_tor'] - df['away_tor']
    df['tord_diff'] = df['home_tord'] - df['away_tord']
    df['orb_diff'] = df['home_orb'] - df['away_orb']
    df['drb_diff'] = df['home_drb'] - df['away_drb']

    mean_rest = df[['home_rest', 'away_rest']].stack().mean()
    missing_home = df['home_rest'].isna().sum()
    missing_away = df['away_rest'].isna().sum()
    df['home_rest'] = df['home_rest'].fillna(mean_rest)
    df['away_rest'] = df['away_rest'].fillna(mean_rest)
    print(f"Imputed {missing_home} home_rest and {missing_away} away_rest with mean ({mean_rest:.1f} days)")

    df['rest_diff'] = df['home_rest'] - df['away_rest']
    df['is_neutral'] = df['neutral_site'].astype(int)

    df['home_days_since_last_game'] = pd.to_numeric(df['home_days_since_last_game'], errors='coerce')
    df['away_days_since_last_game'] = pd.to_numeric(df['away_days_since_last_game'], errors='coerce')
    df['home_games_last_7_days'] = pd.to_numeric(df['home_games_last_7_days'], errors='coerce')
    df['away_games_last_7_days'] = pd.to_numeric(df['away_games_last_7_days'], errors='coerce')

    df['home_days_since_last_game'] = df['home_days_since_last_game'].fillna(df['home_rest'])
    df['away_days_since_last_game'] = df['away_days_since_last_game'].fillna(df['away_rest'])
    df['home_games_last_7_days'] = df['home_games_last_7_days'].fillna(1)
    df['away_games_last_7_days'] = df['away_games_last_7_days'].fillna(1)

    df['days_since_last_game_diff'] = (
        df['home_days_since_last_game'] - df['away_days_since_last_game']
    )
    df['games_last_7_days_diff'] = (
        df['home_games_last_7_days'] - df['away_games_last_7_days']
    )
    df['home_fatigue_index'] = (
        df['home_games_last_7_days'] / np.maximum(df['home_days_since_last_game'], 1.0)
    )
    df['away_fatigue_index'] = (
        df['away_games_last_7_days'] / np.maximum(df['away_days_since_last_game'], 1.0)
    )
    df['fatigue_index_diff'] = df['home_fatigue_index'] - df['away_fatigue_index']

    feature_cols = [
        'adj_o_diff',
        'adj_d_diff',
        'efg_diff',
        'is_neutral',
        'tor_diff',
        'tord_diff',
        'orb_diff',
        'drb_diff',
        'days_since_last_game_diff',
        'games_last_7_days_diff',
        'fatigue_index_diff',
    ]
    
    return df, feature_cols


def build_model(df, feature_cols, torvik, target='margin', use_fixed_cutoff=True):
    """Train and evaluate the Ridge model on a chronological split."""
    model_df = df.dropna(subset=[target] + feature_cols).copy()
    
    if len(model_df) == 0:
        print("No data available after dropping missing values!")
        return None, None, None
    
    model_df['date_dt'] = pd.to_datetime(model_df['date'])
    model_df = model_df.sort_values('date_dt').reset_index(drop=True)

    split_idx = int(len(model_df) * 0.8)
    cutoff_date = model_df.loc[split_idx - 1, 'date_dt']

    if use_fixed_cutoff:
        torvik_cutoff = torvik.copy()
        torvik_cutoff['date_dt'] = pd.to_datetime(torvik_cutoff['date'])
        torvik_cutoff = torvik_cutoff[torvik_cutoff['date_dt'] <= cutoff_date]
        torvik_cutoff = torvik_cutoff.sort_values('date_dt')
        torvik_cutoff = torvik_cutoff.groupby('team_id', as_index=False).tail(1)
        if 'team_id' not in torvik_cutoff.columns:
            torvik_cutoff = torvik_cutoff.reset_index()

        stat_names = []
        torvik_cols = set(torvik_cutoff.columns)
        for feat in feature_cols:
            if not feat.endswith('_diff'):
                continue
            stat = feat[:-5]
            if f'home_{stat}' in model_df.columns and f'away_{stat}' in model_df.columns and stat in torvik_cols:
                stat_names.append(stat)

        future_mask = model_df['date_dt'] > cutoff_date
        if future_mask.any() and stat_names:
            for stat in stat_names:
                home_col = f'home_{stat}'
                away_col = f'away_{stat}'
                snapshot = torvik_cutoff.set_index('team_id')[stat].to_dict()
                model_df.loc[future_mask, home_col] = model_df.loc[future_mask, 'home_team_id'].map(snapshot)
                model_df.loc[future_mask, away_col] = model_df.loc[future_mask, 'away_team_id'].map(snapshot)
                diff_col = f'{stat}_diff'
                if diff_col in model_df.columns:
                    model_df.loc[future_mask, diff_col] = (
                        model_df.loc[future_mask, home_col] - model_df.loc[future_mask, away_col]
                    )

        if (
            future_mask.any()
            and 'fatigue_index_diff' in model_df.columns
            and 'home_games_last_7_days' in model_df.columns
            and 'away_games_last_7_days' in model_df.columns
            and 'home_days_since_last_game' in model_df.columns
            and 'away_days_since_last_game' in model_df.columns
        ):
            model_df.loc[future_mask, 'home_fatigue_index'] = (
                model_df.loc[future_mask, 'home_games_last_7_days']
                / np.maximum(model_df.loc[future_mask, 'home_days_since_last_game'], 1.0)
            )
            model_df.loc[future_mask, 'away_fatigue_index'] = (
                model_df.loc[future_mask, 'away_games_last_7_days']
                / np.maximum(model_df.loc[future_mask, 'away_days_since_last_game'], 1.0)
            )
            model_df.loc[future_mask, 'fatigue_index_diff'] = (
                model_df.loc[future_mask, 'home_fatigue_index']
                - model_df.loc[future_mask, 'away_fatigue_index']
            )

        model_df = model_df.dropna(subset=[target] + feature_cols).reset_index(drop=True)

    train_mask = model_df['date_dt'] <= cutoff_date
    test_mask = model_df['date_dt'] > cutoff_date

    X = model_df[feature_cols].values
    y = model_df[target].values

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"\nTraining set: {len(X_train)} games")
    print(f"Test set: {len(X_test)} games")
    print(f"Fixed cutoff date: {cutoff_date.date()}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    alphas = np.logspace(-3, 3, 50)
    ridge_cv = RidgeCV(alphas=alphas, cv=TimeSeriesSplit(n_splits=5))
    ridge_cv.fit(X_train_scaled, y_train)
    
    best_alpha = ridge_cv.alpha_
    print(f"\nBest alpha (regularization): {best_alpha:.4f}")
    
    model = Ridge(alpha=best_alpha)
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    print(f"\n{'Metric':<20} {'Train':<12} {'Test':<12}")
    print("-"*44)
    print(f"{'MAE':<20} {train_mae:<12.2f} {test_mae:<12.2f}")
    print(f"{'RMSE':<20} {train_rmse:<12.2f} {test_rmse:<12.2f}")
    print(f"{'R²':<20} {train_r2:<12.3f} {test_r2:<12.3f}")
    
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE (Coefficient Magnitude)")
    print("="*50)
    coef_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.coef_
    })
    coef_df['abs_coef'] = np.abs(coef_df['coefficient'])
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    
    print(f"\n{'Feature':<30} {'Coefficient':<12}")
    print("-"*42)
    for _, row in coef_df.head(15).iterrows():
        print(f"{row['feature']:<30} {row['coefficient']:<12.4f}")
    
    test_df = model_df.loc[test_mask].reset_index(drop=True)
    if 'spread' in test_df.columns:
        spread_available = test_df['spread'].notna().values
        if spread_available.sum() > 0:
            spread_mae = mean_absolute_error(
                y_test[spread_available],
                -test_df.loc[spread_available, 'spread']
            )
            print(f"\n{'Vegas Spread MAE (test):':<30} {spread_mae:.2f}")
    
    return model, scaler, model_df


def main():
    print("Loading data...")
    acc_teams, torvik, all_games = load_data()
    acc_team_ids = set(acc_teams['team_id'].astype(int).tolist())
    torvik_team_ids = set(torvik['team_id'].astype(int).tolist())

    games = all_games.copy()
    games['home_team_id'] = games['home_team_id'].astype(int)
    games['away_team_id'] = games['away_team_id'].astype(int)
    mapped_mask = (
        games['home_team_id'].isin(torvik_team_ids) &
        games['away_team_id'].isin(torvik_team_ids)
    )
    excluded_games = (~mapped_mask).sum()
    games = games[mapped_mask].reset_index(drop=True)

    print(
        f"Loaded {len(acc_teams)} ACC teams, {len(torvik)} Torvik ratings, "
        f"{len(all_games)} total D1 games, {len(games)} Torvik-covered games "
        f"(excluded {excluded_games} games with unmapped/non-Torvik teams)"
    )
    
    print("\nPreparing features...")
    df, feature_cols = prepare_features(games, torvik, all_games, acc_team_ids)
    
    print(f"\nFeatures ({len(feature_cols)}):")
    for f in feature_cols:
        print(f"  - {f}")
    
    print("\nBuilding Ridge regression model...")
    print(f"Target: margin (home_score - away_score)")
    
    model, scaler, model_df = build_model(df, feature_cols, torvik, use_fixed_cutoff=True)
    
    if model is not None:
        print("\n" + "="*50)
        print("SAMPLE PREDICTIONS (Last 10 Test Games)")
        print("="*50)
        test_df = model_df.tail(10)
        X_sample = scaler.transform(test_df[feature_cols].values)
        preds = model.predict(X_sample)
        
        print(f"\n{'Date':<12} {'Home':<8} {'Away':<8} {'Actual':<10} {'Predicted':<10} {'Error':<8}")
        print("-"*66)
        for i, (idx, row) in enumerate(test_df.iterrows()):
            actual = row['margin']
            pred = preds[i]
            error = actual - pred
            print(f"{row['date']:<12} {int(row['home_team_id']):<8} {int(row['away_team_id']):<8} {actual:<10.0f} {pred:<10.1f} {error:<8.1f}")


if __name__ == "__main__":
    main()
