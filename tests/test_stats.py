import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

from winlock.stats import (
    add_wilson,
    adjust_pvalues,
    binomial_vs_half,
    diff_z,
    evaluate_predictions,
    log_loss,
    match_log_losses,
    paired_loss_difference,
    wilson_ci,
)


@pytest.mark.parametrize("wins,n", [(11, 20), (1, 3), (520, 1000), (4321, 9000)])
def test_wilson_matches_statsmodels(wins, n):
    low, high = wilson_ci(wins, n)
    ref_low, ref_high = proportion_confint(wins, n, alpha=0.05, method="wilson")
    assert low == pytest.approx(ref_low)
    assert high == pytest.approx(ref_high)


def test_wilson_11_of_20_by_hand():
    # Worked by hand with z = 1.96: center ≈ 0.5426, half ≈ 0.2062.
    low, high = wilson_ci(11, 20)
    assert low == pytest.approx(0.342, abs=1e-3)
    assert high == pytest.approx(0.742, abs=1e-3)


@pytest.mark.parametrize("wins,n", [(0, 10), (10, 10)])
def test_wilson_stays_inside_0_1(wins, n):
    low, high = wilson_ci(wins, n)
    assert 0 <= low < high <= 1


def test_wilson_vectorized_and_narrows_with_n():
    low, high = wilson_ci(np.array([5, 50, 500]), np.array([10, 100, 1000]))
    widths = high - low
    assert np.all(np.diff(widths) < 0)


def test_add_wilson_does_not_modify_input():
    df = pd.DataFrame({"hero_id": [1, 2], "games": [20, 100], "wins": [11, 40]})
    out = add_wilson(df)
    assert list(df.columns) == ["hero_id", "games", "wins"]
    assert out["win_rate"].tolist() == [0.55, 0.40]
    assert (out["ci_low"] < out["win_rate"]).all() and (out["win_rate"] < out["ci_high"]).all()


def test_diff_z_by_hand():
    # 60/100 vs 50/100: SE = sqrt(.6*.4/100 + .5*.5/100) = sqrt(.0049) = 0.07
    diff, z, p = diff_z(60, 100, 50, 100)
    assert diff == pytest.approx(-0.10)
    assert z == pytest.approx(-0.10 / 0.07)
    assert p == pytest.approx(2 * norm.sf(0.10 / 0.07))


def test_holm_is_stricter_than_bh():
    p = np.array([0.001, 0.01, 0.02, 0.04, 0.3])
    holm, bh = adjust_pvalues(p, "holm"), adjust_pvalues(p, "fdr_bh")
    assert np.all(holm >= bh) and np.all(bh >= p)


def test_evaluate_predictions_small_case():
    m = evaluate_predictions([1, 0, 1, 0], [0.9, 0.2, 0.4, 0.5])
    assert m["correct"] == 3  # 0.4 calls Team1 (wrong); 0.5 calls Team1 (right)
    assert m["brier"] == pytest.approx((0.01 + 0.04 + 0.36 + 0.25) / 4)
    assert evaluate_predictions([1, 0], [0.5, 0.5])["log_loss"] == pytest.approx(np.log(2))


def test_paired_loss_difference_identical_models_is_zero_and_better_is_negative():
    y = np.array([1, 0, 1, 0, 1, 1])
    p = np.array([0.6, 0.4, 0.55, 0.45, 0.7, 0.3])
    better = np.where(y == 1, 0.8, 0.2)
    with np.errstate(invalid="ignore", divide="ignore"):
        same = paired_loss_difference(y, p, p)
    assert same["mean_diff"] == 0
    assert paired_loss_difference(y, p, better)["mean_diff"] < 0


def test_log_loss_reference_values():
    assert log_loss([1, 0], [0.5, 0.5]) == pytest.approx(np.log(2))
    assert match_log_losses([1], [0.9])[0] == pytest.approx(-np.log(0.9))
    assert np.isfinite(log_loss([1], [0.0]))  # clipped, not infinite


def test_binomial_vs_half_and_corrections():
    df = pd.DataFrame({"hero_id": [1, 2, 3], "games": [1000, 1000, 20], "wins": [600, 500, 11]})
    out = binomial_vs_half(df)
    assert out.loc[0, "p_vs_half"] < 1e-9  # 600/1000 is far from 50%
    assert out.loc[1, "p_vs_half"] == pytest.approx(1.0)
    assert out.loc[2, "p_vs_half"] > 0.5  # 11/20 can't be told from 50%
    assert (out["p_vs_half_holm"] >= out["p_vs_half_bh"]).all()
