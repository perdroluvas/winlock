"""Pipeline entry point: each command loads data, calls the analysis modules, prints and saves.

One command per part of the article (windows in `[windows]` of config/analysis.toml):
    uv run winlock extract      # read the window from the public data dumps (network)
    uv run winlock quality      # population, window sizes, the dip in match starts at the update
    uv run winlock figures      # Wilson CI per hero, 24 h before and 24 h after the update
    uv run winlock patch        # how did win rates change across the update? (+ placebo)
    uv run winlock model        # logistic regression: before vs after the update, retraining
    uv run winlock portraits    # download hero portraits once for the win-rate chart (network)
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from winlock import evaluation, extract, figures, model, portraits, quality, theme
from winlock.config import load_config, resolve_path
from winlock.db import connect, load_heroes
from winlock.population import (
    add_hero_names,
    create_population,
    hero_winrates,
    population_report,
    window,
    window_sizes,
)
from winlock.stats import add_wilson, binomial_vs_half, evaluate_predictions

BEFORE_AFTER = ("before", "after")


class Run:
    """Shared setup: config, connection with the population table, output folders."""

    def __init__(self) -> None:
        self.cfg: dict[str, Any] = load_config()
        self.con = connect()
        create_population(self.con, self.cfg)
        self.heroes = load_heroes()
        self.alpha = self.cfg["stats"]["alpha"]
        self.figures = resolve_path(self.cfg["figures"]["dir"])
        self.tables = resolve_path(self.cfg["figures"]["tables_dir"])
        self.tables.mkdir(parents=True, exist_ok=True)
        self.dpi = self.cfg["figures"]["dpi"]
        theme.use(self.cfg["figures"]["theme"])
        self.portraits = resolve_path(self.cfg["portraits"]["dir"])
        self.written: list[Path] = []

    def win(self, name: str) -> tuple[str, str]:
        return window(self.cfg, name)

    def label(self, name: str) -> str:
        """'pre (10-29 20:30 – 10-30 20:30 UTC)'."""
        start, end = (t[5:16].replace("T", " ") for t in self.win(name))
        return f"{name} ({start} – {end} UTC)"

    def named(self, rates: pd.DataFrame) -> pd.DataFrame:
        """Add Wilson CIs and hero names to a games/wins table."""
        return add_hero_names(add_wilson(rates, self.alpha), self.heroes)

    def winrates(self, name: str) -> pd.DataFrame:
        return self.named(hero_winrates(self.con, self.win(name)))

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


def print_change(s: dict[str, Any], label: str) -> None:
    print(f"{label}: {s['heroes']} heroes")
    print(f"  SD of z: {s['sd_z']:.2f} (1.00 if only binomial noise)")
    print(f"  correlation of hero rates: {s['corr']:.2f}")
    print(
        f"  significant changes: {s['significant_raw']} raw, {s['significant_holm']} Holm, "
        f"{s['significant_bh']} BH\n"
    )


def compare(run: Run, before: str, after: str) -> pd.DataFrame:
    """Per-hero before/after comparison of two named windows."""
    a, b = (run.winrates(n) for n in (before, after))
    return evaluation.compare_samples(a, b, BEFORE_AFTER)


def by_abs(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Rows sorted by |col|, largest first."""
    return df.reindex(df[col].abs().sort_values(ascending=False).index)


def cmd_quality() -> None:
    run = Run()
    pop, patch, ex = run.cfg["population"], run.cfg["patch"], run.cfg["extract"]
    show("Rows and matches per UTC start day:", quality.start_day_profile(run.con, 4), 1)
    show(
        f"Rows per match, {pop['game_mode']} mode (whole extract):",
        quality.match_completeness(run.con, ex["window_start"], ex["window_end"], pop["game_mode"]),
        2,
    )
    show("Population (cumulative filters):", population_report(run.con, run.cfg), 1)
    print(f"Excluded flags: {', '.join(pop.get('exclude_flags', [])) or 'none'}\n")
    show("Complete matches per window (start_time, UTC):", window_sizes(run.con, run.cfg))

    minutes = patch["dip_bin_minutes"]
    starts = quality.starts_per_bin(run.con, *patch["dip_window"], pop["game_mode"], minutes)
    show(f"{pop['game_mode']} starts per {minutes} min around the update:", starts)
    run.chart(
        figures.patch_dip_chart,
        starts,
        run.figures / "quality_patch_dip.png",
        "The update in the data: match starts dip",
        update_at=pd.Timestamp(patch["published_at"]),
        buffer=tuple(pd.Timestamp(t) for t in patch["buffer"]),
    )
    run.report_written()


def cmd_figures() -> None:
    run = Run()
    for name, when in (("pre", "24 h before the update"), ("post", "24 h after the update")):
        rates = binomial_vs_half(run.winrates(name))
        run.table(rates, f"winrates_{name}.csv")
        excludes = ((rates["ci_low"] > 0.5) | (rates["ci_high"] < 0.5)).sum()
        lo, hi = rates.loc[rates["win_rate"].idxmin()], rates.loc[rates["win_rate"].idxmax()]
        print(
            f"{run.label(name)}: {len(rates)} heroes, win rates {lo['win_rate']:.1%} "
            f"({lo['name']}) to {hi['win_rate']:.1%} ({hi['name']}); different from 50%: "
            f"{excludes} by their own CI, {(rates['p_vs_half_holm'] < run.alpha).sum()} after "
            f"Holm, {(rates['p_vs_half_bh'] < run.alpha).sum()} after BH"
        )
        run.chart(
            figures.winrate_ci_chart,
            rates,
            run.figures / f"winrate_ci_{name}.png",
            f"Hero win rates, {when}",
            alpha=run.alpha,
            portraits_dir=run.portraits,
        )
    print()
    run.report_written()


def cmd_patch() -> None:
    run = Run()
    results: dict[str, pd.DataFrame] = {}
    for comparison, c in run.cfg["comparisons"].items():
        ev = compare(run, c["before"], c["after"])
        results[comparison] = ev
        run.table(ev, f"{comparison}_changes.csv")
        label = f"{comparison}: {run.label(c['before'])} vs {run.label(c['after'])}"
        print_change(evaluation.change_summary(ev, BEFORE_AFTER, run.alpha), label)

    cols = ["name", "games_before", "rate_before", "games_after", "rate_after", "diff", "z"]
    show(
        "Largest changes across the update:",
        by_abs(results["patch"], "z").head(10)[cols + ["p_holm"]],
    )

    run.chart(
        figures.before_after_chart,
        results["patch"],
        run.figures / "patch_before_after.png",
        "Hero win rates before vs after the update",
    )
    run.report_written()


def cmd_model() -> None:
    run = Run()
    hero_ids = run.heroes["hero_id"].tolist()
    pre = model.load_match_data(run.con, hero_ids, run.win("train"), run.win("pre"))
    post = model.match_data(hero_ids, pre.train, model.match_table(run.con, run.win("post")))
    print(
        f"Matches: train {len(pre.y_train):,}, test before {len(pre.y_test):,}, "
        f"test after {len(post.y_test):,}"
    )
    print(
        f"Team0 win share: train {pre.y_train.mean():.1%}, before {pre.y_test.mean():.1%}, "
        f"after {post.y_test.mean():.1%}\n"
    )

    train_rates = run.winrates("train").set_index("hero_id")["win_rate"]
    rates = model.rate_vector(train_rates, hero_ids)
    preds_pre, logreg, val_scores = model.fit_baselines(pre, rates, run.cfg)
    show("Validation (last 20% of train) log loss per C:", val_scores, 5)
    print(f"Chosen C = {logreg.C}\n")
    preds_post = model.predict_baselines(post.X_test, pre.y_train.mean(), rates, logreg)

    rows = []
    for when, y, preds in (("before", pre.y_test, preds_pre), ("after", post.y_test, preds_post)):
        for name, p in preds.items():
            rows.append(
                {"model": f"{name}, {when}", "test": when, **evaluate_predictions(y, p, run.alpha)}
            )
    metrics = pd.DataFrame(rows)
    run.table(metrics, "model_metrics.csv")
    show("Test metrics, 24 h before vs after the update (coin flip: log loss 0.693):", metrics, 4)

    check = model.retrain_check(post, logreg, run.cfg)
    run.table(check["metrics"], "model_retrain.csv")
    print(
        f"Retrain check: refit on the first {check['n_fit']:,} post-update matches; both models "
        f"scored on the last {check['n_eval']:,} (from {check['eval_start']:%m-%d %H:%M} UTC)"
    )
    show("", check["metrics"], 4)
    d = check["loss_diff"]
    print(
        f"Log loss, retrained − old: {d['mean_diff']:+.5f} (95% CI {d['ci_low']:+.5f} to "
        f"{d['ci_high']:+.5f}, p = {d['p']:.3g}); negative = retraining helps\n"
    )

    coef = add_hero_names(pd.DataFrame({"hero_id": hero_ids, "coef": logreg.coef_[0]}), run.heroes)
    coef = coef.merge(train_rates.rename("rate_train"), left_on="hero_id", right_index=True)
    run.table(coef.sort_values("coef"), "model_coefficients.csv")
    corr = np.corrcoef(coef["rate_train"], coef["coef"])[0, 1]
    print(f"Correlation between coefficient and train win rate: {corr:.2f}\n")

    run.chart(
        figures.model_accuracy_chart,
        metrics,
        run.figures / "model_accuracy.png",
        "Old-game model, before vs after the update",
    )
    run.chart(
        figures.model_accuracy_chart,
        check["metrics"],
        run.figures / "model_retrain.png",
        "After the update: old vs retrained model",
    )
    lr = "Logistic regression"
    run.chart(
        figures.calibration_chart,
        {
            "before the update": (pre.y_test, preds_pre[lr]),
            "after the update": (post.y_test, preds_post[lr]),
        },
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


def cmd_portraits() -> None:
    cfg = load_config()
    p = cfg["portraits"]
    dest = resolve_path(p["dir"])
    hero_ids = load_heroes()["hero_id"].tolist()
    written = portraits.fetch_portraits(p["api_url"], p["image_key"], dest, hero_ids, p["size_px"])
    cached = sum((dest / f"{h}.png").exists() for h in hero_ids)
    print(f"Downloaded {len(written)} portraits; {cached} of {len(hero_ids)} heroes cached")


def cmd_extract(dry_run: bool = False) -> None:
    cfg = load_config()
    meta = extract.run_extract(cfg, resolve_path(cfg["extract"]["out_dir"]), dry_run=dry_run)
    gap = meta["ingestion_gap_hours"]
    print(
        f"Manifest v{meta['manifest_version']} ({meta['manifest_generated_at']}), "
        f"generation {meta['generation']}, ingested up to {meta['watermark_hi']}"
    )
    print(f"Window {meta['window_start']} -> {meta['window_end']}")
    if gap > 0:
        print(f"  WARNING: the last {gap:.1f} h of the window are not ingested yet")
    print("  (late arrivals: rerun a day or two after the window ends to complete it)")
    probes = ", ".join(f"{p}: {t[5:16]}" for p, t in meta["partition_starts_probed"].items())
    print(f"Partition starts probed (footers only): {probes}")
    print(f"Partitions holding the window: {meta['partitions']}")
    print(
        f"Files selected: {len(meta['files'])} {meta['files_by_kind']}, "
        f"{meta['files_gb']} GB whole (only the needed columns/row groups are read)"
    )
    if dry_run:
        print("Dry run: nothing read.")
        return
    s = meta["summary"]
    print(
        f"Rows read: {meta['rows_raw']:,}; after deduplication: {meta['rows_final']:,} "
        f"({meta['duplicates_removed']:,} duplicates removed)"
    )
    print(
        f"Matches: {s['matches']:,}; start {s['start_min']} -> {s['start_max']}; "
        f"created {s['created_min']} -> {s['created_max']}"
    )
    print(f"Wrote {Path(meta['output']).relative_to(resolve_path('.'))} (+ meta.json)")


COMMANDS = {
    "extract": (cmd_extract, "read the patch window from the public data dumps (network)"),
    "quality": (cmd_quality, "population, window sizes, the dip in match starts at the update"),
    "figures": (cmd_figures, "Wilson CI per hero, 24 h before and after the update"),
    "patch": (cmd_patch, "win rates before vs after the update, plus the placebo"),
    "model": (cmd_model, "logistic regression: before vs after the update, and retraining"),
    "portraits": (cmd_portraits, "download hero portraits once (network) for the win-rate chart"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="winlock analysis pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in COMMANDS.items():
        sub.add_parser(name, help=help_text)
    sub.choices["extract"].add_argument(
        "--dry-run", action="store_true", help="only list the files that would be read"
    )
    args = vars(parser.parse_args())
    COMMANDS[args.pop("command")][0](**args)


if __name__ == "__main__":
    main()
