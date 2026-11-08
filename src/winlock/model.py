"""Match-outcome models: can the heroes in a match predict who wins?

Each match becomes one row: a +1 for every hero on Team0, a −1 for every hero on Team1, and the
target "Team0 won". If hero win rates are real signal, models fit on train beat a constant
guess on test; if they are noise, they don't.
"""

from dataclasses import dataclass
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from winlock.population import Window, in_window
from winlock.stats import evaluate_predictions, log_loss, paired_loss_difference

SIDE_ONLY = "No heroes (side only)"


def match_table(con: duckdb.DuckDBPyConnection, win: Window) -> pd.DataFrame:
    """One row per match started in the window: start time, hero lists per team, Team0 won."""
    return con.execute(
        f"""
        SELECT match_id,
               MIN(start_time) AS start_time,
               LIST(hero_id) FILTER (WHERE team = 'Team0') AS team0_heroes,
               LIST(hero_id) FILTER (WHERE team = 'Team1') AS team1_heroes,
               BOOL_OR(won AND team = 'Team0') AS team0_won
        FROM population
        WHERE {in_window(win)}
        GROUP BY match_id
        ORDER BY start_time, match_id
        """
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
    fit, val = np.flatnonzero(fit), np.flatnonzero(val)  # boolean masks -> row indices
    scores = []
    for C in cfg["model"]["C_grid"]:
        m = LogisticRegression(C=C, max_iter=1000, random_state=seed).fit(X[fit], y[fit])
        scores.append({"C": C, "val_log_loss": log_loss(y[val], m.predict_proba(X[val])[:, 1])})
    scores = pd.DataFrame(scores)
    best_C = scores.loc[scores["val_log_loss"].idxmin(), "C"]
    final = LogisticRegression(C=best_C, max_iter=1000, random_state=seed).fit(X, y)
    return final, scores


@dataclass
class MatchData:
    """Train (fit) and test matches with their hero features and targets (Team0 won = 1)."""

    hero_ids: list[int]
    train: pd.DataFrame
    test: pd.DataFrame
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_match_data(
    con: duckdb.DuckDBPyConnection, hero_ids: list[int], fit_window: Window, test_window: Window
) -> MatchData:
    """Build the match tables, hero matrices and targets: fit on one window, test on another."""
    return match_data(hero_ids, match_table(con, fit_window), match_table(con, test_window))


def match_data(hero_ids: list[int], train: pd.DataFrame, test: pd.DataFrame) -> MatchData:
    """MatchData from two match tables (see match_table)."""
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
    preds = predict_baselines(data.X_test, data.y_train.mean(), rates, logreg)
    return preds, logreg, val_scores


def predict_baselines(
    X_test: np.ndarray, team0_share: float, rates: np.ndarray, logreg: LogisticRegression
) -> dict[str, np.ndarray]:
    """Predictions of the three fitted models for any hero matrix (e.g. another test window)."""
    return {
        SIDE_ONLY: np.full(len(X_test), team0_share),
        "Win-rate model": winrate_model(X_test, rates),
        "Logistic regression": logreg.predict_proba(X_test)[:, 1],
    }


def retrain_check(
    post: MatchData, old_model: LogisticRegression, cfg: dict[str, Any]
) -> dict[str, Any]:
    """After the update: does a model refit on new-game matches beat the old one?

    `post.test` is the post-patch window. Its first `post_fit_fraction` (in time) refits a
    logistic regression (C chosen again on the last part of that slice); both models then
    predict **the same** remaining matches, so the comparison is paired match by match.
    Returns the metrics of both models and the paired log-loss difference (new − old; negative =
    retraining helps).
    """
    alpha = cfg["stats"]["alpha"]
    n_fit = round(len(post.test) * cfg["model"]["post_fit_fraction"])
    fit_rows, eval_rows = np.arange(n_fit), np.arange(n_fit, len(post.test))
    X_fit, y_fit = post.X_test[fit_rows], post.y_test[fit_rows]
    X_eval, y_eval = post.X_test[eval_rows], post.y_test[eval_rows]
    fit, val = temporal_holdout(post.test.iloc[fit_rows], cfg["model"]["validation_fraction"])
    new_model, val_scores = fit_logistic(X_fit, y_fit, fit, val, cfg)
    p_old = old_model.predict_proba(X_eval)[:, 1]
    p_new = new_model.predict_proba(X_eval)[:, 1]
    metrics = pd.DataFrame(
        [
            {"model": "Trained before the update", **evaluate_predictions(y_eval, p_old, alpha)},
            {"model": "Retrained after the update", **evaluate_predictions(y_eval, p_new, alpha)},
        ]
    )
    return {
        "metrics": metrics,
        "loss_diff": paired_loss_difference(y_eval, p_old, p_new, alpha),
        "val_scores": val_scores,
        "model": new_model,
        "n_fit": n_fit,
        "n_eval": len(eval_rows),
        "eval_start": post.test["start_time"].iloc[n_fit],
    }
