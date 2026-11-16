"""Did the win rates change? Compare the same heroes in two samples."""

from typing import Any

import numpy as np
import pandas as pd

from winlock.stats import adjust_pvalues, diff_z

RATE_COLUMNS = ["hero_id", "games", "wins", "win_rate", "ci_low", "ci_high"]


def compare_samples(a: pd.DataFrame, b: pd.DataFrame, names: tuple[str, str]) -> pd.DataFrame:
    """Per hero present in both win-rate tables: did the rate change from a to b?

    Columns of a and b get the suffixes _<names[0]> and _<names[1]>, with win_rate renamed
    to rate_<name>. Adds diff (b − a), z, p, p_holm and p_bh. Sorted by z, index reset.
    """
    raise NotImplementedError


def change_summary(ev: pd.DataFrame, names: tuple[str, str], alpha: float) -> dict[str, Any]:
    """sd_z ≈ 1: only noise. Well above 1: the rates really changed.

    Returns heroes, sd_z (sample SD), corr (between the two rate columns),
    significant_raw, significant_holm and significant_bh (counts below alpha).
    """
    raise NotImplementedError


def extract_agreement(old: pd.DataFrame, new: pd.DataFrame, alpha: float) -> dict[str, Any]:
    """The same before/after comparison on two extracts of the same window: what moved?

    Returns heroes, max_rate_shift (largest |new − old| of any before or after rate),
    max_diff_shift, corr_diff, significant_old, significant_new and significant_both
    (Holm, below alpha).
    """
    raise NotImplementedError
