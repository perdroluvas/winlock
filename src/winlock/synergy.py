"""Teammate synergy: do some hero pairs win more than their individual strength predicts?

Synergy is measured against the additive win-rate model: per pair, the wins it actually got
together minus the wins the model expected from the 12 heroes in each of those matches.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from winlock.model import (
    MatchData,
    fit_logistic,
    pair_matrix,
    team_pair_outcomes,
    temporal_holdout,
    winrate_model,
)
from winlock.stats import adjust_pvalues, evaluate_predictions, pair_synergy, paired_loss_difference


def pair_tables(
    data: MatchData, rates: np.ndarray, names: pd.Series, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-pair synergy, discovered on train and checked on test.

    Returns (train_pairs, pairs):
    - train_pairs: the train result, tested pairs only (`synergy.min_pair_matches`), with
      names and Holm/BH-adjusted p-values over all tested pairs;
    - pairs: train_pairs joined with the same pairs on test (suffixes _train/_test).
    """
    alpha = cfg["stats"]["alpha"]
    syn = {
        split: pair_synergy(team_pair_outcomes(matches, winrate_model(X, rates)), alpha)
        for split, matches, X in (
            ("train", data.train, data.X_train),
            ("test", data.test, data.X_test),
        )
    }
    train_pairs = syn["train"][syn["train"]["matches"] >= cfg["synergy"]["min_pair_matches"]]
    train_pairs = train_pairs.assign(
        pair=lambda d: names[d["hero_a"]].values + " + " + names[d["hero_b"]].values,
        p_holm=lambda d: adjust_pvalues(d["p"], "holm"),
        p_bh=lambda d: adjust_pvalues(d["p"], "fdr_bh"),
    ).reset_index(drop=True)
    pairs = train_pairs.merge(
        syn["test"], on=["hero_a", "hero_b"], how="left", suffixes=("_train", "_test")
    )
    return train_pairs, pairs.sort_values("p_train").reset_index(drop=True)


def replication(pairs: pd.DataFrame) -> dict[str, float]:
    """Correlation of the pairs' synergy z between train and test.

    If synergy were pure noise the two would be unrelated: correlation 0 ± 1.96/sqrt(n).
    """
    both = pairs.dropna(subset=["z_test"])
    return {
        "pairs": len(both),
        "corr": float(np.corrcoef(both["z_train"], both["z_test"])[0, 1]),
        "noise_band": float(1.96 / np.sqrt(len(both))),
    }


def compare_pair_model(data: MatchData, cfg: dict[str, Any]) -> dict[str, Any]:
    """Logistic regression with heroes only vs heroes + teammate pairs, scored on test.

    Both pick C on the same temporal validation slice. The models are compared match by match
    (paired log-loss difference), which is more sensitive than comparing two accuracies.
    """
    alpha = cfg["stats"]["alpha"]
    fit, val = temporal_holdout(data.train, cfg["model"]["validation_fraction"])
    base, base_scores = fit_logistic(data.X_train, data.y_train, fit, val, cfg)
    full_train = sparse.hstack(
        [sparse.csr_matrix(data.X_train), pair_matrix(data.train, data.hero_ids)]
    ).tocsr()
    full_test = sparse.hstack(
        [sparse.csr_matrix(data.X_test), pair_matrix(data.test, data.hero_ids)]
    ).tocsr()
    full, full_scores = fit_logistic(full_train, data.y_train, fit, val, cfg)
    p_base = base.predict_proba(data.X_test)[:, 1]
    p_full = full.predict_proba(full_test)[:, 1]
    metrics = pd.DataFrame(
        [
            {
                "model": "Heroes only",
                "C": base.C,
                **evaluate_predictions(data.y_test, p_base, alpha),
            },
            {
                "model": "Heroes + pairs",
                "C": full.C,
                **evaluate_predictions(data.y_test, p_full, alpha),
            },
        ]
    )
    return {
        "metrics": metrics,
        "val_scores": base_scores.merge(full_scores, on="C", suffixes=("_heroes", "_pairs")),
        "loss_diff": paired_loss_difference(data.y_test, p_base, p_full, alpha),
    }
