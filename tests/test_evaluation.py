import pandas as pd
import pytest

from winlock.evaluation import change_summary, compare_samples
from winlock.stats import add_wilson

NAMES = ("before", "after")


def rates(wins):
    """Win-rate table for heroes 1..k, 1000 games each."""
    return add_wilson(
        pd.DataFrame({"hero_id": range(1, len(wins) + 1), "games": 1000, "wins": wins})
    )


def test_compare_samples_columns_and_drops_missing_heroes():
    a, b = rates([500, 550, 450]), rates([500, 560])  # hero 3 missing in b
    ev = compare_samples(a, b, NAMES)
    assert sorted(ev["hero_id"]) == [1, 2]
    for col in ["rate_before", "rate_after", "diff", "z", "p", "p_holm", "p_bh"]:
        assert col in ev
    assert ev.loc[ev.hero_id == 1, "z"].item() == pytest.approx(0)


def test_identical_samples_have_no_change():
    a = rates([480, 500, 520, 540])
    s = change_summary(compare_samples(a, a, NAMES), NAMES, alpha=0.05)
    assert s["heroes"] == 4 and s["sd_z"] == 0
    assert s["corr"] == pytest.approx(1.0)
    assert s["significant_holm"] == 0


def test_a_big_change_is_significant():
    a, b = rates([500, 500, 500]), rates([500, 500, 600])  # hero 3: 50% -> 60%
    s = change_summary(compare_samples(a, b, NAMES), NAMES, alpha=0.05)
    assert s["significant_holm"] == 1


def test_extract_agreement_of_identical_extracts():
    from winlock.evaluation import extract_agreement

    ev = pd.DataFrame(
        {
            "hero_id": [1, 2, 3],
            "rate_before": [0.50, 0.52, 0.47],
            "rate_after": [0.50, 0.49, 0.48],
            "diff": [0.0, -0.03, 0.01],
            "p_holm": [1.0, 0.001, 0.4],
        }
    )
    same = extract_agreement(ev, ev, alpha=0.05)
    assert same["max_rate_shift"] == 0 and same["max_diff_shift"] == 0
    assert same["significant_old"] == same["significant_new"] == same["significant_both"] == 1

    shifted = ev.assign(rate_after=ev["rate_after"] + 0.002, diff=ev["diff"] + 0.002)
    out = extract_agreement(ev, shifted, alpha=0.05)
    assert out["max_rate_shift"] == pytest.approx(0.002)
    assert out["max_diff_shift"] == pytest.approx(0.002)
