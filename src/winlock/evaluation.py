"""Do the win rates hold? Compare the same heroes in two samples, hero by hero."""

from typing import Any

import numpy as np
import pandas as pd

from winlock.stats import (
    adjust_pvalues,
    coverage,
    diff_z,
    expected_coverage,
    simulate_coverage,
)

RATE_COLUMNS = ["hero_id", "games", "wins", "win_rate", "ci_low", "ci_high"]


def compare_samples(a: pd.DataFrame, b: pd.DataFrame, names: tuple[str, str]) -> pd.DataFrame:
    """Per hero: rates in both samples, the change b − a, its z, p and Holm/BH-adjusted p.

    `a` and `b` are win-rate tables with Wilson CIs (see stats.add_wilson); heroes missing from
    either sample are dropped. Columns get the sample names as suffixes, e.g. rate_train.
    """
    sa, sb = (f"_{n}" for n in names)
    ev = a.merge(b[RATE_COLUMNS], on="hero_id", suffixes=(sa, sb))
    ev = ev.rename(columns={f"win_rate{sa}": f"rate{sa}", f"win_rate{sb}": f"rate{sb}"})
    ev["diff"], ev["z"], ev["p"] = diff_z(
        ev[f"wins{sa}"], ev[f"games{sa}"], ev[f"wins{sb}"], ev[f"games{sb}"]
    )
    ev["p_holm"] = adjust_pvalues(ev["p"], "holm")
    ev["p_bh"] = adjust_pvalues(ev["p"], "fdr_bh")
    ev[f"inside{sa}_ci"] = (ev[f"ci_low{sa}"] <= ev[f"rate{sb}"]) & (
        ev[f"rate{sb}"] <= ev[f"ci_high{sa}"]
    )
    return ev.sort_values("z").reset_index(drop=True)


def stability_summary(
    ev: pd.DataFrame, names: tuple[str, str], cfg: dict[str, Any]
) -> dict[str, Any]:
    """How stable were the rates between the two samples, compared with pure chance?

    - coverage: share of heroes whose b-rate falls inside their a-CI, next to the value expected
      if nothing changed (formula and binomial simulation; it is below 95% because b is noisy too)
    - sd_z: 1.0 if the changes are pure binomial noise
    - corr: correlation of the hero rates between the samples
    """
    sa, sb = names
    alpha = cfg["stats"]["alpha"]
    sims = simulate_coverage(
        ev[f"rate_{sa}"],
        ev[f"games_{sa}"],
        ev[f"games_{sb}"],
        alpha,
        cfg["evaluation"]["n_sims"],
        cfg["stats"]["seed"],
    )
    observed = coverage(ev[f"rate_{sb}"], ev[f"ci_low_{sa}"], ev[f"ci_high_{sa}"])
    lo, hi = np.quantile(sims, [0.025, 0.975])
    return {
        "heroes": len(ev),
        "coverage": observed,
        "coverage_formula": float(
            np.mean(expected_coverage(ev[f"games_{sa}"], ev[f"games_{sb}"], alpha))
        ),
        "coverage_sim_mean": float(sims.mean()),
        "coverage_sim_low": float(lo),
        "coverage_sim_high": float(hi),
        "share_sims_at_or_below": float(np.mean(sims <= observed)),
        "sd_z": float(np.std(ev["z"], ddof=1)),
        "corr": float(np.corrcoef(ev[f"rate_{sa}"], ev[f"rate_{sb}"])[0, 1]),
        "significant_raw": int((ev["p"] < alpha).sum()),
        "significant_holm": int((ev["p_holm"] < alpha).sum()),
        "significant_bh": int((ev["p_bh"] < alpha).sum()),
        "sims": sims,
    }
