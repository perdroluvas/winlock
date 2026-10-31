import numpy as np
import pytest

from winlock.model import (
    hero_matrix,
    hero_pairs,
    match_table,
    pair_matrix,
    team_pair_outcomes,
    temporal_holdout,
    winrate_model,
)


@pytest.fixture
def train_matches(ten_match_con):
    return match_table(ten_match_con, "train")


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


def test_hero_pairs_count():
    assert len(hero_pairs(list(range(44)))) == 946


def test_pair_matrix_has_15_pairs_per_team(train_matches):
    X = pair_matrix(train_matches, hero_ids=list(range(1, 13)))
    assert X.shape == (7, 66)  # 12 heroes -> 66 pairs
    dense = X.toarray()
    assert ((dense == 1).sum(axis=1) == 15).all()  # pairs inside Team0 (heroes 1-6)
    assert ((dense == -1).sum(axis=1) == 15).all()  # pairs inside Team1 (heroes 7-12)
    assert (dense.sum(axis=1) == 0).all()


def test_team_pair_outcomes_rows_and_expectations(train_matches):
    out = team_pair_outcomes(train_matches, np.full(len(train_matches), 0.7))
    assert len(out) == 7 * 30
    team0_pair = out[(out.hero_a == 1) & (out.hero_b == 2)]
    assert team0_pair["won"].all() and np.allclose(team0_pair["expected"], 0.7)
    team1_pair = out[(out.hero_a == 7) & (out.hero_b == 8)]
    assert (~team1_pair["won"]).all() and np.allclose(team1_pair["expected"], 0.3)


def test_load_match_data_and_rate_vector(ten_match_con):
    import pandas as pd

    from winlock.model import load_match_data, rate_vector

    data = load_match_data(ten_match_con, list(range(1, 13)))
    assert data.X_train.shape == (7, 12) and data.X_test.shape == (3, 12)
    assert data.y_train.tolist() == [1.0] * 7
    rates = rate_vector(pd.Series({1: 0.6}), [1, 2])
    assert rates.tolist() == [0.6, 0.5]
