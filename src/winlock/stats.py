"""Statistics for win rates and predictions. Pure functions: inputs -> outputs."""

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from statsmodels.stats.multitest import multipletests


def wilson_ci(wins, n, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for wins/n: always inside [0, 1], unlike p ± z·SE.

    wins and n may be numbers or arrays. Returns (low, high).
    """
    raise NotImplementedError


def match_log_losses(y, p) -> np.ndarray:
    """Log loss of each prediction p (probability that y = 1), one value per match.

    p is clipped to [1e-12, 1 − 1e-12] so the logarithm stays finite.
    """
    raise NotImplementedError


def log_loss(y, p) -> float:
    """Mean of match_log_losses, as a float."""
    raise NotImplementedError


def add_wilson(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Copy of df (columns wins, games) with win_rate, ci_low and ci_high added."""
    raise NotImplementedError


def binomial_vs_half(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df with p_vs_half (exact two-sided binomial test of wins out of games
    against 0.5), p_vs_half_holm and p_vs_half_bh.
    """
    raise NotImplementedError


def diff_z(wins_a, n_a, wins_b, n_b) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Change b − a and its z. Both samples are noisy: SE = sqrt(SE_a² + SE_b²).

    Returns (diff, z, two-sided p).
    """
    raise NotImplementedError


def adjust_pvalues(p, method: str) -> np.ndarray:
    """Holm (no false alarm at all) or BH (few false alarms) adjusted p-values.

    method is a statsmodels name: "holm" or "fdr_bh". Output keeps the input order.
    """
    raise NotImplementedError


def evaluate_predictions(y, p, alpha: float = 0.05) -> dict[str, float]:
    """Score predictions p against outcomes y.

    A call is correct when p > 0.5 and y = 1, or p <= 0.5 and y = 0. Returns a dict with
    n, correct, accuracy, acc_ci_low, acc_ci_high (Wilson), log_loss and brier.
    """
    raise NotImplementedError


def paired_loss_difference(y, p_a, p_b, alpha: float = 0.05) -> dict[str, float]:
    """Per-match log-loss difference (b − a) on the same matches; negative = b is better.

    Returns a dict with mean_diff, ci_low, ci_high, z and p (two-sided).
    """
    raise NotImplementedError
