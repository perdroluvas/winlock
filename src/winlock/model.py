"""Match-outcome models: can the heroes in a match predict who wins?

Each match becomes one row: a +1 for every hero on Team0, a −1 for every hero on Team1, and the
target "Team0 won". If hero win rates are real signal, models fit on train beat a constant
guess on test; if they are noise, they don't.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from winlock.stats import log_loss

SIDE_ONLY = "No heroes (side only)"


def match_table(con: duckdb.DuckDBPyConnection, split: str) -> pd.DataFrame:
    """One row per match of a split: start time, hero lists per team and whether Team0 won."""
    return con.execute(
        """
        SELECT match_id,
               MIN(start_time) AS start_time,
               LIST(hero_id) FILTER (WHERE team = 'Team0') AS team0_heroes,
               LIST(hero_id) FILTER (WHERE team = 'Team1') AS team1_heroes,
               BOOL_OR(won AND team = 'Team0') AS team0_won
        FROM population
        WHERE split = $split
        GROUP BY match_id
        ORDER BY start_time, match_id
        """,
        {"split": split},
    ).df()


def hero_matrix(matches: pd.DataFrame, hero_ids: list[int]) -> np.ndarray:
    """Features: column j is +1 if hero_ids[j] played for Team0, −1 for Team1, 0 if absent.

    Columns come from the fixed hero list, so train and test always line up, even when a hero
    (e.g. one released that day) only appears on one side.
    """
    col = {h: j for j, h in enumerate(hero_ids)}
    X = np.zeros((len(matches), len(hero_ids)))
    for i, (team0, team1) in enumerate(
        zip(matches["team0_heroes"], matches["team1_heroes"], strict=True)
    ):
        for h in team0:
            X[i, col[h]] += 1
        for h in team1:
            X[i, col[h]] -= 1
    return X


def temporal_holdout(matches: pd.DataFrame, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (fit, validation): the last `fraction` of matches in time is validation."""
    is_val = np.arange(len(matches)) >= round(len(matches) * (1 - fraction))
    return ~is_val, is_val


def winrate_model(X: np.ndarray, hero_rates: np.ndarray) -> np.ndarray:
    """P(Team0 wins) from the train win rates alone: σ(Σ logit(rate) Team0 − Σ logit(rate) Team1).

    No fitting: it only adds up how much each hero won in train. Unknown heroes have rate 0.5,
    so logit = 0 and they count for nothing.
    """
    return expit(X @ logit(hero_rates))


def fit_logistic(
    X, y: np.ndarray, fit: np.ndarray, val: np.ndarray, cfg: dict[str, Any]
) -> tuple[LogisticRegression, pd.DataFrame]:
    """Choose the L2 strength C on the temporal validation slice, then refit on all of train.

    Returns the final model and the validation log loss of each C.
    """
    seed = cfg["stats"]["seed"]
    fit, val = np.flatnonzero(fit), np.flatnonzero(val)  # row indices work for dense and sparse
    scores = []
    for C in cfg["model"]["C_grid"]:
        m = LogisticRegression(C=C, max_iter=1000, random_state=seed).fit(X[fit], y[fit])
        scores.append({"C": C, "val_log_loss": log_loss(y[val], m.predict_proba(X[val])[:, 1])})
    scores = pd.DataFrame(scores)
    best_C = scores.loc[scores["val_log_loss"].idxmin(), "C"]
    final = LogisticRegression(C=best_C, max_iter=1000, random_state=seed).fit(X, y)
    return final, scores


def hero_pairs(hero_ids: list[int]) -> list[tuple[int, int]]:
    """All unordered teammate pairs (a, b) with a < b: 44 heroes give 44·43/2 = 946 pairs."""
    return list(combinations(sorted(hero_ids), 2))


def pair_matrix(matches: pd.DataFrame, hero_ids: list[int]) -> sparse.csr_matrix:
    """Pair features: column k is +1 if pair k played together on Team0, −1 on Team1, else 0.

    Each team of 6 holds 15 pairs, so every row has 30 non-zero entries out of 946: a sparse
    matrix stores only those.
    """
    col = {pair: k for k, pair in enumerate(hero_pairs(hero_ids))}
    rows, cols, vals = [], [], []
    for i, (team0, team1) in enumerate(
        zip(matches["team0_heroes"], matches["team1_heroes"], strict=True)
    ):
        for team, sign in ((team0, 1.0), (team1, -1.0)):
            for pair in combinations(sorted(team), 2):
                rows.append(i)
                cols.append(col[pair])
                vals.append(sign)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(matches), len(col)))


def team_pair_outcomes(matches: pd.DataFrame, p_team0: np.ndarray) -> pd.DataFrame:
    """One row per (match, team, teammate pair): did the team win, and what did the additive
    model expect (`p_team0` for Team0, 1 − p_team0 for Team1)?
    """
    hero_a, hero_b, won, expected = [], [], [], []
    for team0, team1, team0_won, p0 in zip(
        matches["team0_heroes"],
        matches["team1_heroes"],
        matches["team0_won"],
        p_team0,
        strict=True,
    ):
        for team, team_won, p in ((team0, team0_won, p0), (team1, not team0_won, 1 - p0)):
            for a, b in combinations(sorted(team), 2):
                hero_a.append(a)
                hero_b.append(b)
                won.append(team_won)
                expected.append(p)
    return pd.DataFrame({"hero_a": hero_a, "hero_b": hero_b, "won": won, "expected": expected})


@dataclass
class MatchData:
    """Train and test matches with their hero features and targets (Team0 won = 1)."""

    hero_ids: list[int]
    train: pd.DataFrame
    test: pd.DataFrame
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_match_data(con: duckdb.DuckDBPyConnection, hero_ids: list[int]) -> MatchData:
    """Build the match tables, hero matrices and targets for both splits."""
    train, test = match_table(con, "train"), match_table(con, "test")
    return MatchData(
        hero_ids=hero_ids,
        train=train,
        test=test,
        X_train=hero_matrix(train, hero_ids),
        X_test=hero_matrix(test, hero_ids),
        y_train=train["team0_won"].to_numpy(dtype=float),
        y_test=test["team0_won"].to_numpy(dtype=float),
    )


def rate_vector(train_rates: pd.Series, hero_ids: list[int]) -> np.ndarray:
    """Train win rate per hero in `hero_ids` order; heroes never seen in train count as 50%."""
    return train_rates.reindex(hero_ids).fillna(0.5).to_numpy()


def fit_baselines(
    data: MatchData, rates: np.ndarray, cfg: dict[str, Any]
) -> tuple[dict[str, np.ndarray], LogisticRegression, pd.DataFrame]:
    """Fit the three heroes-only models on train and return their test predictions.

    1. Side only: always the train share of Team0 wins.
    2. Win-rate model: train win rates added up as log-odds (no fitting).
    3. Logistic regression, C chosen on the last part of train (in time).
    Returns the predictions by model name, the logistic model and its validation scores.
    """
    fit, val = temporal_holdout(data.train, cfg["model"]["validation_fraction"])
    logreg, val_scores = fit_logistic(data.X_train, data.y_train, fit, val, cfg)
    preds = {
        SIDE_ONLY: np.full(len(data.y_test), data.y_train.mean()),
        "Win-rate model": winrate_model(data.X_test, rates),
        "Logistic regression": logreg.predict_proba(data.X_test)[:, 1],
    }
    return preds, logreg, val_scores
