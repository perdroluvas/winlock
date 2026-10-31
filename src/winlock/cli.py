"""Pipeline entry point: each command loads data, calls the analysis modules, prints and saves.

Usage:
    uv run winlock quality      # data checks: start days, start hours, complete matches
    uv run winlock figures      # population report + train-side baseline charts
    uv run winlock evaluate     # do the train win rates hold on test? + split-half control
    uv run winlock model        # can the heroes in a match predict the winner on test?
    uv run winlock synergy      # do some teammate pairs win more than their parts predict?
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from winlock import evaluation, figures, model, quality, synergy
from winlock.config import load_config, resolve_path
from winlock.db import connect, load_heroes
from winlock.population import (
    add_hero_names,
    create_population,
    hero_winrates,
    hero_winrates_by_parity,
    population_report,
    split_summary,
)
from winlock.stats import add_wilson, binomial_vs_half, evaluate_predictions


class Run:
    """Shared setup: config, connection with the population table, output folders."""

    def __init__(self, population: bool = True) -> None:
        self.cfg: dict[str, Any] = load_config()
        self.con = connect()
        if population:
            create_population(self.con, self.cfg)
        self.heroes = load_heroes()
        self.alpha = self.cfg["stats"]["alpha"]
        self.figures = resolve_path(self.cfg["figures"]["dir"])
        self.tables = resolve_path(self.cfg["figures"]["tables_dir"])
        self.tables.mkdir(parents=True, exist_ok=True)
        self.dpi = self.cfg["figures"]["dpi"]
        self.day = self.cfg["population"]["start_day"]
        self.written: list[Path] = []

    def named(self, rates: pd.DataFrame) -> pd.DataFrame:
        """Add Wilson CIs and hero names to a games/wins table."""
        return add_hero_names(add_wilson(rates, self.alpha), self.heroes)

    def winrates(self, split: str) -> pd.DataFrame:
        return self.named(hero_winrates(self.con, split))

    def table(self, df: pd.DataFrame, name: str) -> None:
        df.to_csv(self.tables / name, index=False)
        self.written.append(self.tables / name)

    def chart(self, fn, *args, **kwargs) -> None:
        self.written.append(fn(*args, dpi=self.dpi, **kwargs))

    def report_written(self) -> None:
        root = resolve_path(".")
        print("Wrote:", *[p.relative_to(root) for p in self.written], sep="\n  ")


def show(title: str, df: pd.DataFrame, digits: int = 3) -> None:
    numeric = df.select_dtypes("number").columns
    print(f"{title}\n{df.round({c: digits for c in numeric}).to_string(index=False)}\n")


def print_stability(s: dict[str, Any], label: str) -> None:
    print(f"{label}: {s['heroes']} heroes")
    print(f"  coverage (second rate inside first 95% CI): observed {s['coverage']:.1%}")
    print(
        f"  if nothing changed: formula {s['coverage_formula']:.1%}, simulation "
        f"{s['coverage_sim_mean']:.1%} (95% of simulations {s['coverage_sim_low']:.1%}–"
        f"{s['coverage_sim_high']:.1%}; {s['share_sims_at_or_below']:.1%} at or below observed)"
    )
    print(f"  SD of z: {s['sd_z']:.2f} (1.00 if only binomial noise)")
    print(f"  correlation of hero rates: {s['corr']:.2f}")
    print(
        f"  significant changes: {s['significant_raw']} raw, {s['significant_holm']} Holm, "
        f"{s['significant_bh']} BH\n"
    )


def cmd_quality() -> None:
    run = Run(population=False)
    pop = run.cfg["population"]
    show("Rows and matches per UTC start day (whole file):", quality.start_day_profile(run.con), 1)
    show(f"Start hours on {run.day} (UTC):", quality.start_hour_profile(run.con, run.day))
    show(
        f"Rows per match on {run.day}, {pop['game_mode']} mode:",
        quality.match_completeness(run.con, run.day, pop["game_mode"]),
        2,
    )


def cmd_figures() -> None:
    run = Run()
    show("Population (cumulative filters):", population_report(run.con, run.cfg), 1)
    show("Split:", split_summary(run.con))

    train = binomial_vs_half(run.winrates("train"))
    run.table(train, "train_winrates.csv")
    excludes = ((train["ci_low"] > 0.5) | (train["ci_high"] < 0.5)).sum()
    print(
        f"Heroes different from 50%: {excludes} of {len(train)} by their own CI, "
        f"{(train['p_vs_half_holm'] < run.alpha).sum()} after Holm, "
        f"{(train['p_vs_half_bh'] < run.alpha).sum()} after BH\n"
    )

    title = f"train (70%), {run.day}"
    run.chart(
        figures.winrate_ci_chart,
        train,
        run.figures / "train_winrate_ci.png",
        f"Hero win rates, {title}",
        alpha=run.alpha,
    )
    run.chart(
        figures.games_per_hero_chart,
        train,
        run.figures / "train_games_per_hero.png",
        f"Games per hero, {title}",
    )
    run.chart(
        figures.ci_width_vs_games_chart,
        train,
        run.figures / "train_ci_width_vs_games.png",
        "More games, narrower interval",
        alpha=run.alpha,
    )
    run.report_written()


def cmd_evaluate() -> None:
    run = Run()
    train, test = run.winrates("train"), run.winrates("test")
    only_test = sorted(set(test["name"]) - set(train["name"]))
    if only_test:
        print(f"Only in test (no train CI, not evaluated): {', '.join(only_test)}\n")

    # Temporal: morning (train) vs evening (test).
    ev = evaluation.compare_samples(train, test, ("train", "test"))
    temporal = evaluation.stability_summary(ev, ("train", "test"), run.cfg)
    run.table(ev, "evaluation.csv")
    print_stability(temporal, "Train (00:00–13:35) vs test (13:35–22:00)")

    # Split-half control: even vs odd match_id over the whole day, so no time difference.
    even, odd = (run.named(hero_winrates_by_parity(run.con, k)) for k in (0, 1))
    halves = evaluation.compare_samples(even, odd, ("even", "odd"))
    split_half = evaluation.stability_summary(halves, ("even", "odd"), run.cfg)
    run.table(halves, "evaluation_split_half.csv")
    print_stability(split_half, "Split-half control: even vs odd match_id (same hours)")

    cols = ["name", "games_train", "rate_train", "games_test", "rate_test", "z", "p_holm"]
    top = ev.reindex(ev["z"].abs().sort_values(ascending=False).index).head(8)
    show("Largest train→test changes:", top[cols])

    run.chart(
        figures.train_vs_test_chart,
        ev,
        run.figures / "eval_train_vs_test.png",
        "Did the win rates hold? Train vs test",
    )
    run.chart(
        figures.z_distribution_chart,
        {"train → test": ev["z"], "even vs odd (control)": halves["z"]},
        run.figures / "eval_z_distribution.png",
        "Changes between samples vs pure chance",
    )
    run.chart(
        figures.coverage_simulation_chart,
        temporal["sims"],
        temporal["coverage"],
        run.figures / "eval_coverage_simulation.png",
        "How often does the test rate land in the train CI?",
    )
    run.report_written()


def cmd_model() -> None:
    run = Run()
    hero_ids = run.heroes["hero_id"].tolist()
    data = model.load_match_data(run.con, hero_ids)
    print(f"Matches: train {len(data.y_train):,}, test {len(data.y_test):,}")
    print(f"Team0 win share: train {data.y_train.mean():.1%}, test {data.y_test.mean():.1%}\n")

    train_rates = run.winrates("train").set_index("hero_id")["win_rate"]
    preds, logreg, val_scores = model.fit_baselines(
        data, model.rate_vector(train_rates, hero_ids), run.cfg
    )
    show("Validation (last 20% of train) log loss per C:", val_scores, 5)
    print(f"Chosen C = {logreg.C}\n")

    metrics = pd.DataFrame(
        [
            {"model": name, **evaluate_predictions(data.y_test, p, run.alpha)}
            for name, p in preds.items()
        ]
    )
    run.table(metrics, "model_metrics.csv")
    show("Test metrics (coin flip: log loss 0.693, Brier 0.250):", metrics, 4)

    coef = add_hero_names(pd.DataFrame({"hero_id": hero_ids, "coef": logreg.coef_[0]}), run.heroes)
    coef = coef.merge(train_rates.rename("rate_train"), left_on="hero_id", right_index=True)
    run.table(coef.sort_values("coef"), "model_coefficients.csv")
    corr = np.corrcoef(coef["rate_train"], coef["coef"])[0, 1]
    print(f"Correlation between coefficient and train win rate: {corr:.2f}\n")

    run.chart(
        figures.model_accuracy_chart,
        metrics,
        run.figures / "model_accuracy.png",
        "Predicting the winner of test matches",
    )
    run.chart(
        figures.calibration_chart,
        {k: (data.y_test, p) for k, p in preds.items() if k != model.SIDE_ONLY},
        run.figures / "model_calibration.png",
        "Are the predicted probabilities honest?",
    )
    run.chart(
        figures.coefficients_chart,
        coef,
        run.figures / "model_coefficients.png",
        "What the model learned vs the train win rates",
    )
    run.report_written()


def cmd_synergy() -> None:
    run = Run()
    hero_ids = run.heroes["hero_id"].tolist()
    data = model.load_match_data(run.con, hero_ids)
    train_rates = run.winrates("train").set_index("hero_id")["win_rate"]
    names = run.heroes.set_index("hero_id")["name"]

    # 1. Per pair: discovered on train, checked on test.
    train_pairs, pairs = synergy.pair_tables(
        data, model.rate_vector(train_rates, hero_ids), names, run.cfg
    )
    run.table(pairs, "synergy_pairs.csv")
    n_pairs, m = len(train_pairs), train_pairs["matches"]
    print(f"Teammate pairs tested: {n_pairs} of {len(model.hero_pairs(hero_ids))} possible")
    print(
        f"Matches together per pair: median {m.median():.0f}, "
        f"< 100: {(m < 100).sum()} pairs, >= 1,000: {(m >= 1000).sum()} pairs"
    )
    print(
        f"p < {run.alpha} before correction: {(train_pairs['p'] < run.alpha).sum()} "
        f"(about {run.alpha * n_pairs:.0f} expected by pure chance); "
        f"after Holm: {(train_pairs['p_holm'] < run.alpha).sum()}, "
        f"after BH: {(train_pairs['p_bh'] < run.alpha).sum()}\n"
    )
    cols = [
        "pair",
        "matches_train",
        "win_rate_train",
        "expected_rate_train",
        "excess_train",
        "z_train",
        "p_holm",
        "matches_test",
        "excess_test",
        "z_test",
    ]
    show("Strongest pairs on train, and the same pairs on test:", pairs.head(10)[cols])
    show("Your example:", pairs[pairs["pair"] == "Rat King + Violet"][cols])
    rep = synergy.replication(pairs)
    print(
        f"Replication: correlation of synergy z, train vs test = {rep['corr']:.3f} over "
        f"{rep['pairs']} pairs (pure noise: 0 ± {rep['noise_band']:.3f})\n"
    )

    # 2. Model: do the pair terms help predict test matches?
    cmp = synergy.compare_pair_model(data, run.cfg)
    run.table(cmp["metrics"], "model_metrics_synergy.csv")
    show("Validation log loss per C:", cmp["val_scores"], 5)
    show("Test metrics:", cmp["metrics"], 4)
    d = cmp["loss_diff"]
    print(
        f"Log loss change from adding pairs: {d['mean_diff']:+.5f} "
        f"(95% CI {d['ci_low']:+.5f} to {d['ci_high']:+.5f}, p = {d['p']:.3f}); "
        "negative = pairs help\n"
    )

    run.chart(
        figures.pair_matches_chart,
        train_pairs,
        run.figures / "synergy_pair_matches.png",
        "Matches together per teammate pair",
    )
    run.chart(
        figures.synergy_volcano_chart,
        train_pairs,
        run.figures / "synergy_volcano.png",
        f"Teammate synergy on train: {n_pairs} tests at once",
        alpha=run.alpha,
    )
    run.chart(
        figures.synergy_replication_chart,
        pairs.dropna(subset=["z_test"]),
        run.figures / "synergy_replication.png",
        "Does a pair's synergy replicate on test?",
    )
    run.chart(
        figures.model_accuracy_chart,
        cmp["metrics"],
        run.figures / "synergy_model_accuracy.png",
        "Do pair terms improve predictions?",
    )
    run.report_written()


COMMANDS = {
    "quality": (cmd_quality, "data checks: start days, start hours, complete matches"),
    "figures": (cmd_figures, "population report + train baseline charts"),
    "evaluate": (cmd_evaluate, "train vs test win rates + split-half control"),
    "model": (cmd_model, "predict match winners from heroes, scored on test"),
    "synergy": (cmd_synergy, "teammate pair synergy: per-pair tests + model with pair terms"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="winlock analysis pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in COMMANDS.items():
        sub.add_parser(name, help=help_text)
    args = parser.parse_args()
    COMMANDS[args.command][0]()


if __name__ == "__main__":
    main()
