import numpy as np
import pandas as pd
import pytest

from winlock.evaluation import compare_samples, stability_summary
from winlock.stats import add_wilson

CFG = {"stats": {"alpha": 0.05, "seed": 0}, "evaluation": {"n_sims": 500}}


def rates(wins):
    """Win-rate table for heroes 1..k, 1000 games each."""
    return add_wilson(
        pd.DataFrame({"hero_id": range(1, len(wins) + 1), "games": 1000, "wins": wins})
    )


def test_compare_samples_columns_and_drops_missing_heroes():
    a, b = rates([500, 550, 450]), rates([500, 560])  # hero 3 missing in b
    ev = compare_samples(a, b, ("train", "test"))
    assert sorted(ev["hero_id"]) == [1, 2]
    for col in ["rate_train", "rate_test", "z", "p", "p_holm", "p_bh", "inside_train_ci"]:
        assert col in ev
    assert ev.loc[ev.hero_id == 1, "z"].item() == pytest.approx(0)


def test_identical_samples_are_perfectly_stable():
    a = rates([480, 500, 520, 540])
    s = stability_summary(compare_samples(a, a, ("x", "y")), ("x", "y"), CFG)
    assert s["coverage"] == 1.0
    assert s["sd_z"] == 0
    assert s["corr"] == pytest.approx(1.0)
    assert s["significant_holm"] == 0
    assert len(s["sims"]) == 500


@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in divide"
)  # constant rates: corr = NaN
def test_formula_coverage_for_equal_samples():
    a = rates([500] * 5)
    s = stability_summary(compare_samples(a, a, ("x", "y")), ("x", "y"), CFG)
    assert s["coverage_formula"] == pytest.approx(0.834, abs=1e-3)
    assert np.isclose(s["coverage_sim_mean"], 0.834, atol=0.03)
