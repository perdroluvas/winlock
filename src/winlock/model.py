"""Logistic regression: can the 12 heroes of a match predict the winner?"""

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
    """One row per population match in the window, ordered by start_time then match_id.

    Columns: match_id, start_time, team0_heroes and team1_heroes (lists of hero_id),
    team0_won.
    """
    raise NotImplementedError


def hero_matrix(matches: pd.DataFrame, hero_ids: list[int]) -> np.ndarray:
    """+1 for each hero on Team0, −1 for each hero on Team1, 0 if absent.

    One row per match, one column per entry of hero_ids, in that order.
    """
    raise NotImplementedError


def temporal_holdout(matches: pd.DataFrame, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (fit, val) over matches already in time order.

    val is the last `fraction` of the rows; fit is everything before it.
    """
    raise NotImplementedError


def winrate_model(X: np.ndarray, hero_rates: np.ndarray) -> np.ndarray:
    """P(Team0 wins) from train win rates alone, no fitting: σ(Σ logit(rate) Team0 − Team1)."""
    raise NotImplementedError


def fit_logistic(
    X, y: np.ndarray, fit: np.ndarray, val: np.ndarray, cfg: dict[str, Any]
) -> tuple[LogisticRegression, pd.DataFrame]:
    """Choose C on the validation slice (later in time), then refit on all of train.

    fit and val are boolean masks. For each C in cfg["model"]["C_grid"], fit on the fit
    rows and score log loss on the val rows. Returns (model refit on every row at the best
    C, DataFrame with columns C and val_log_loss).
    """
    raise NotImplementedError


@dataclass
class MatchData:
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
    """MatchData for a fit window and a test window, read from the population table."""
    raise NotImplementedError


def match_data(hero_ids: list[int], train: pd.DataFrame, test: pd.DataFrame) -> MatchData:
    """Bundle two match tables with their hero matrices and outcomes (team0_won as floats)."""
    raise NotImplementedError


def rate_vector(train_rates: pd.Series, hero_ids: list[int]) -> np.ndarray:
    """Win rates aligned to hero_ids, as an array; a hero with no rate gets 0.5."""
    raise NotImplementedError


def fit_baselines(
    data: MatchData, rates: np.ndarray, cfg: dict[str, Any]
) -> tuple[dict[str, np.ndarray], LogisticRegression, pd.DataFrame]:
    """Fit the logistic regression on train and predict the test matches with all three models.

    Returns (predictions as in predict_baselines, the fitted model, the validation scores).
    """
    raise NotImplementedError


def predict_baselines(
    X_test: np.ndarray, team0_share: float, rates: np.ndarray, logreg: LogisticRegression
) -> dict[str, np.ndarray]:
    """P(Team0 wins) for each test match, per model.

    Keys: SIDE_ONLY (the train Team0 win share for every match), "Win-rate model" and
    "Logistic regression".
    """
    raise NotImplementedError


def retrain_check(
    post: MatchData, old_model: LogisticRegression, cfg: dict[str, Any]
) -> dict[str, Any]:
    """Refit on early post-update matches; score old and new model on the same later ones.

    The first cfg["model"]["post_fit_fraction"] of post.test (in time) refits the model, C
    chosen as in fit_logistic; the rest scores both models. Returns a dict with metrics (one
    row per model: "Trained before the update", "Retrained after the update"), loss_diff
    (paired, new − old), val_scores, model, n_fit, n_eval and eval_start.
    """
    raise NotImplementedError


def bootstrap_coefficients(
    X: np.ndarray, y: np.ndarray, C: float, n_refits: int, seed: int
) -> np.ndarray:
    """Refit on matches drawn with replacement; one row of weights per refit.

    The unit drawn is the whole match, so the spread of a weight across refits shows how
    much it depends on which matches happened to be in train.
    """
    raise NotImplementedError


def coefficient_intervals(boot: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval per weight: the middle 1 − alpha of the refits.

    boot has one row per refit. Returns (low, high), one value per weight.
    """
    raise NotImplementedError
