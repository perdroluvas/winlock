"""Smoke tests: every chart renders from a tiny table and writes a non-empty PNG."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from winlock import figures, theme
from winlock.stats import add_wilson, binomial_vs_half


@pytest.fixture(autouse=True, params=sorted(theme.THEMES))
def each_theme(request):
    """Every chart test runs once per theme."""
    before = theme.active().name
    yield theme.use(request.param)
    theme.use(before)


@pytest.fixture
def heroes():
    df = pd.DataFrame({"hero_id": [1, 2, 3], "games": [900, 1500, 400], "wins": [500, 700, 200]})
    out = binomial_vs_half(add_wilson(df))
    out["name"] = ["Infernus", "Seven", "Vindicta"]
    return out


@pytest.fixture
def ev(heroes):
    return heroes.assign(
        rate_train=heroes["win_rate"],
        rate_test=heroes["win_rate"] + [0.01, -0.02, 0.0],
        z=[0.5, -1.2, 0.0],
        p_holm=[1.0, 0.6, 1.0],
    )


def check(path):
    assert path.exists() and path.stat().st_size > 1000


def test_baseline_charts(heroes, tmp_path):
    check(figures.winrate_ci_chart(heroes, tmp_path / "a.png", "t"))
    check(figures.games_per_hero_chart(heroes, tmp_path / "b.png", "t"))
    check(figures.ci_width_vs_games_chart(heroes, tmp_path / "c.png", "t"))


def test_winrate_chart_with_portraits(heroes, tmp_path):
    # Portraits for two of the three heroes: the third keeps just its name.
    for hero_id in (1, 2):
        Image.new("RGBA", (16, 16), (200, 120, 10, 255)).save(tmp_path / f"{hero_id}.png")
    check(figures.winrate_ci_chart(heroes, tmp_path / "p.png", "t", portraits_dir=tmp_path))


def test_evaluation_charts(ev, tmp_path):
    check(figures.train_vs_test_chart(ev, tmp_path / "a.png", "t"))
    zs = {"a": pd.Series([0.1, -0.5, 1.2]), "b": pd.Series([0.3, 0.0, -1.0])}
    check(figures.z_distribution_chart(zs, tmp_path / "b.png", "t"))
    check(
        figures.coverage_simulation_chart(np.array([0.6, 0.7, 0.75]), 0.8, tmp_path / "c.png", "t")
    )


def test_model_charts(ev, tmp_path):
    metrics = pd.DataFrame(
        {
            "model": ["a", "b"],
            "accuracy": [0.5, 0.6],
            "acc_ci_low": [0.48, 0.58],
            "acc_ci_high": [0.52, 0.62],
        }
    )
    check(figures.model_accuracy_chart(metrics, tmp_path / "a.png", "t"))
    rng = np.random.default_rng(0)
    p = rng.uniform(0.3, 0.7, 200)
    y = (rng.uniform(size=200) < p).astype(float)
    check(figures.calibration_chart({"m": (y, p)}, tmp_path / "b.png", "t"))
    coef = ev.assign(coef=[0.1, -0.2, 0.0])
    check(figures.coefficients_chart(coef, tmp_path / "c.png", "t", n_outliers=1))


def test_synergy_charts(tmp_path):
    pairs = pd.DataFrame(
        {
            "pair": ["A + B", "A + C", "B + C"],
            "matches": [100, 800, 3000],
            "excess": [0.05, -0.02, 0.01],
            "p": [0.01, 0.4, 0.7],
            "p_holm": [0.03, 0.8, 0.8],
            "z_train": [2.5, -0.8, 0.4],
            "z_test": [1.0, -0.2, 0.1],
        }
    )
    check(figures.pair_matches_chart(pairs, tmp_path / "a.png", "t"))
    check(figures.synergy_volcano_chart(pairs, tmp_path / "b.png", "t", n_labels=2))
    check(figures.synergy_replication_chart(pairs, tmp_path / "c.png", "t"))
