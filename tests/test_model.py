import numpy as np
import pandas as pd
import pytest

from conftest import TEST, TRAIN
from winlock.model import (
    hero_matrix,
    match_table,
    temporal_holdout,
    winrate_model,
)


@pytest.fixture
def train_matches(ten_match_con):
    return match_table(ten_match_con, TRAIN)


def test_match_table_has_one_row_per_match(train_matches):
    assert len(train_matches) == 7
    assert train_matches["start_time"].is_monotonic_increasing
    assert train_matches["team0_won"].all()  # in the fixture Team0 always wins


def test_hero_matrix_is_plus_minus_one_per_team(train_matches):
    X = hero_matrix(train_matches, hero_ids=list(range(1, 13)))
    # fixture: heroes 1–6 on Team0, 7–12 on Team1
    assert (X[:, :6] == 1).all() and (X[:, 6:] == -1).all()
    assert (X.sum(axis=1) == 0).all()  # 6 vs 6


def test_hero_matrix_unknown_column_stays_zero(train_matches):
    X = hero_matrix(train_matches, hero_ids=list(range(1, 13)) + [87])
    assert (X[:, -1] == 0).all()


def test_temporal_holdout_takes_the_end(train_matches):
    fit, val = temporal_holdout(train_matches, fraction=0.2)
    assert fit.tolist() == [True] * 6 + [False]  # round(7 * 0.8) = 6
    assert (fit ^ val).all()


def test_winrate_model_neutral_rates_give_half():
    X = np.array([[1, 1, -1, -1], [1, -1, 1, -1]])
    assert np.allclose(winrate_model(X, np.full(4, 0.5)), 0.5)


def test_winrate_model_favours_the_stronger_team():
    X = np.array([[1, -1]])  # hero 0 vs hero 1
    assert winrate_model(X, np.array([0.6, 0.4]))[0] > 0.5


def test_load_match_data_and_rate_vector(ten_match_con):
    from winlock.model import load_match_data, rate_vector

    data = load_match_data(ten_match_con, list(range(1, 13)), TRAIN, TEST)
    assert data.X_train.shape == (7, 12) and data.X_test.shape == (3, 12)
    assert data.y_train.tolist() == [1.0] * 7
    rates = rate_vector(pd.Series({1: 0.6}), [1, 2])
    assert rates.tolist() == [0.6, 0.5]


def synthetic(n, seed, strength=1.0):
    """n fake matches of 4 heroes (2 vs 2) where hero 0 is strong and hero 3 is weak."""
    rng = np.random.default_rng(seed)
    team0 = [sorted(rng.choice(4, 2, replace=False).tolist()) for _ in range(n)]
    team1 = [sorted(set(range(4)) - set(t)) for t in team0]
    power = np.array([strength, 0.0, 0.0, -strength])
    p = 1 / (1 + np.exp(-(power[np.array(team0)].sum(1) - power[np.array(team1)].sum(1))))
    return pd.DataFrame(
        {
            "start_time": pd.date_range("2026-10-30", periods=n, freq="min"),
            "team0_heroes": team0,
            "team1_heroes": team1,
            "team0_won": rng.random(n) < p,
        }
    )


CFG = {
    "stats": {"seed": 0, "alpha": 0.05},
    "model": {"C_grid": [0.1, 1.0], "validation_fraction": 0.2, "post_fit_fraction": 0.7},
}


def test_retrain_check_scores_both_models_on_the_same_late_matches():
    from winlock.model import fit_baselines, match_data, rate_vector, retrain_check

    hero_ids = [0, 1, 2, 3]
    # old game: hero 0 strong; new game: the "patch" flips it, hero 0 is weak
    old = match_data(hero_ids, synthetic(2000, 1), synthetic(10, 2))
    post = match_data(hero_ids, old.train, synthetic(2000, 3, strength=-1.0))
    _, old_model, _ = fit_baselines(old, rate_vector(pd.Series(dtype=float), hero_ids), CFG)
    check = retrain_check(post, old_model, CFG)
    assert check["n_fit"] == 1400 and check["n_eval"] == 600
    assert check["metrics"]["n"].tolist() == [600, 600]
    assert check["loss_diff"]["mean_diff"] < 0  # the refit model learned the flipped game
