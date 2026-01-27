"""
Ridge regression model for ACC basketball game predictions.
"""

import csv
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data():
    """Load game and rating data."""
    games = pd.read_csv(os.path.join("data", "acc_games.csv"))
    torvik = pd.read_csv(os.path.join("data", "torvik_asof_ratings.csv"))
    return games, torvik


def main():
    games, torvik = load_data()
    print(f"Loaded {len(games)} games and {len(torvik)} Torvik ratings")
    print(f"\nGames columns: {list(games.columns)}")
    print(f"Torvik columns: {list(torvik.columns)}")


if __name__ == "__main__":
    main()
