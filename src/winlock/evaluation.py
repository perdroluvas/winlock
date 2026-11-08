"""Did the win rates change? Compare the same heroes in two samples, hero by hero."""

from typing import Any

import numpy as np
import pandas as pd

from winlock.stats import adjust_pvalues, diff_z

RATE_COLUMNS = ["hero_id", "games", "wins", "win_rate", "ci_low", "ci_high"]


def compare_samples(a: pd.DataFrame, b: pd.DataFrame, names: tuple[str, str]) -> pd.DataFrame:
    """Per hero: rates in both samples, the change b − a, its z, p and Holm/BH-adjusted p.

    `a` and `b` are win-rate tables with Wilson CIs (see stats.add_wilson); heroes missing from
    either sample are dropped. Columns get the sample names as suffixes, e.g. rate_before.
    """
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
    """How much did the rates move between the two samples, compared with pure chance?

    - sd_z: 1.0 if the changes are only binomial noise; well above 1 = real change
    - corr: correlation of the hero rates between the samples
    - significant_*: heroes whose change passes alpha raw, after Holm and after BH
    """
    sa, sb = names
    return {
        "heroes": len(ev),
        "sd_z": float(np.std(ev["z"], ddof=1)),
        "corr": float(np.corrcoef(ev[f"rate_{sa}"], ev[f"rate_{sb}"])[0, 1]),
        "significant_raw": int((ev["p"] < alpha).sum()),
        "significant_holm": int((ev["p_holm"] < alpha).sum()),
        "significant_bh": int((ev["p_bh"] < alpha).sum()),
    }
