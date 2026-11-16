"""Did the win rates change? Compare the same heroes in two samples."""

from typing import Any

import numpy as np
import pandas as pd

from winlock.stats import adjust_pvalues, diff_z

RATE_COLUMNS = ["hero_id", "games", "wins", "win_rate", "ci_low", "ci_high"]


def compare_samples(a: pd.DataFrame, b: pd.DataFrame, names: tuple[str, str]) -> pd.DataFrame:
    sa, sb = (f"_{n}" for n in names)
    ev = a.merge(b[RATE_COLUMNS], on="hero_id", suffixes=(sa, sb))
    ev = ev.rename(columns={f"win_rate{sa}": f"rate{sa}", f"win_rate{sb}": f"rate{sb}"})
    ev["diff"], ev["z"], ev["p"] = diff_z(
        ev[f"wins{sa}"], ev[f"games{sa}"], ev[f"wins{sb}"], ev[f"games{sb}"]
    )
    ev["p_holm"] = adjust_pvalues(ev["p"], "holm")
    ev["p_bh"] = adjust_pvalues(ev["p"], "fdr_bh")
    return ev.sort_values("z").reset_index(drop=True)


def change_summary(ev: pd.DataFrame, names: tuple[str, str], alpha: float) -> dict[str, Any]:
    """sd_z ≈ 1: only noise. Well above 1: the rates really changed."""
    sa, sb = names
    return {
        "heroes": len(ev),
        "sd_z": float(np.std(ev["z"], ddof=1)),
        "corr": float(np.corrcoef(ev[f"rate_{sa}"], ev[f"rate_{sb}"])[0, 1]),
        "significant_raw": int((ev["p"] < alpha).sum()),
        "significant_holm": int((ev["p_holm"] < alpha).sum()),
        "significant_bh": int((ev["p_bh"] < alpha).sum()),
    }


def extract_agreement(old: pd.DataFrame, new: pd.DataFrame, alpha: float) -> dict[str, Any]:
    """The same before/after comparison on two extracts of the same window: what moved?"""
    m = old.merge(new, on="hero_id", suffixes=("_old", "_new"))
    sig_old, sig_new = (m[f"p_holm_{k}"] < alpha for k in ("old", "new"))
    return {
        "heroes": len(m),
        "max_rate_shift": float(
            max(
                (m[f"rate_{side}_new"] - m[f"rate_{side}_old"]).abs().max()
                for side in ("before", "after")
            )
        ),
        "max_diff_shift": float((m["diff_new"] - m["diff_old"]).abs().max()),
        "corr_diff": float(np.corrcoef(m["diff_old"], m["diff_new"])[0, 1]),
        "significant_old": int(sig_old.sum()),
        "significant_new": int(sig_new.sum()),
        "significant_both": int((sig_old & sig_new).sum()),
    }
