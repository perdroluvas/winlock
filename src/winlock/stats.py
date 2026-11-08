"""Statistics for win rates. Pure functions: inputs -> outputs, no I/O."""

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from statsmodels.stats.multitest import multipletests


def wilson_ci(wins, n, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a proportion wins/n (works on scalars or arrays).

    center = (p + z²/2n) / (1 + z²/n)
    half   = z * sqrt(p(1-p)/n + z²/4n²) / (1 + z²/n)
    Unlike the normal interval p ± z*sqrt(p(1-p)/n), it never leaves [0, 1] and stays sensible
    for small n or p close to 0 or 1.
    """
    wins = np.asarray(wins, dtype=float)
    n = np.asarray(n, dtype=float)
    z = norm.ppf(1 - alpha / 2)
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def match_log_losses(y, p) -> np.ndarray:
    """Log loss of each prediction: −[y·log(p) + (1−y)·log(1−p)].

    A confident wrong call (p = 0.9, team lost) costs −log(0.1) ≈ 2.3; a 50/50 call always
    costs log(2) ≈ 0.693. p is clipped away from 0 and 1 so a single sure mistake isn't infinite.
    """
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def log_loss(y, p) -> float:
    """Mean log loss over all predictions (lower is better)."""
    return float(match_log_losses(y, p).mean())


def add_wilson(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Return a copy of a `games`/`wins` table with win_rate, ci_low and ci_high columns."""
    out = df.copy()
    out["win_rate"] = out["wins"] / out["games"]
    out["ci_low"], out["ci_high"] = wilson_ci(out["wins"], out["games"], alpha)
    return out


def binomial_vs_half(df: pd.DataFrame) -> pd.DataFrame:
    """Add, per row of a `games`/`wins` table, an exact binomial test of "true rate = 50%",
    plus Holm and BH adjusted p-values over all rows (one test per hero).

    The Wilson CI excluding 50% is the same question for one hero alone; with many heroes read at
    once, only the adjusted p-values keep the overall false-alarm rate at alpha.
    """
    out = df.copy()
    out["p_vs_half"] = [
        binomtest(int(w), int(n), 0.5).pvalue
        for w, n in zip(out["wins"], out["games"], strict=True)
    ]
    out["p_vs_half_holm"] = adjust_pvalues(out["p_vs_half"], "holm")
    out["p_vs_half_bh"] = adjust_pvalues(out["p_vs_half"], "fdr_bh")
    return out


def diff_z(wins_a, n_a, wins_b, n_b) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Difference between two proportions (b − a), its z-score and two-sided p-value.

    z = (p_b − p_a) / sqrt(SE_a² + SE_b²): both samples are noisy, so their variances add.
    """
    p_a = np.asarray(wins_a, dtype=float) / np.asarray(n_a, dtype=float)
    p_b = np.asarray(wins_b, dtype=float) / np.asarray(n_b, dtype=float)
    se = np.sqrt(p_a * (1 - p_a) / np.asarray(n_a) + p_b * (1 - p_b) / np.asarray(n_b))
    diff = p_b - p_a
    z = diff / se
    return diff, z, 2 * norm.sf(np.abs(z))


def adjust_pvalues(p, method: str) -> np.ndarray:
    """Multiple-comparison adjusted p-values: 'holm' (family-wise) or 'fdr_bh' (false discovery)."""
    return multipletests(np.asarray(p, dtype=float), method=method)[1]


def evaluate_predictions(y, p, alpha: float = 0.05) -> dict[str, float]:
    """Score predicted probabilities p for binary outcomes y.

    accuracy: share of matches where the side with p > 0.5 won, with a Wilson CI (it is a
    proportion: correct calls out of n matches). log_loss and brier punish confident mistakes;
    a constant 0.5 scores log(2) ≈ 0.693 and 0.25.
    """
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    correct = int(((p > 0.5) == (y == 1)).sum())
    low, high = wilson_ci(correct, len(y), alpha)
    return {
        "n": len(y),
        "correct": correct,
        "accuracy": correct / len(y),
        "acc_ci_low": float(low),
        "acc_ci_high": float(high),
        "log_loss": log_loss(y, p),
        "brier": float(np.mean((p - y) ** 2)),
    }


def paired_loss_difference(y, p_a, p_b, alpha: float = 0.05) -> dict[str, float]:
    """Mean per-match log-loss difference (b − a) with a normal CI.

    Both models predict the same matches, so comparing them match by match removes the shared
    noise of "which matches happened to be in test"; negative = model b is better.
    """
    d = match_log_losses(y, p_b) - match_log_losses(y, p_a)
    se = d.std(ddof=1) / np.sqrt(len(d))
    z = norm.ppf(1 - alpha / 2)
    return {
        "mean_diff": float(d.mean()),
        "ci_low": float(d.mean() - z * se),
        "ci_high": float(d.mean() + z * se),
        "z": float(d.mean() / se),
        "p": float(2 * norm.sf(abs(d.mean() / se))),
    }
