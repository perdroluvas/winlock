"""Charts for the post. Each function takes an aggregate table and writes one PNG.

Tables must carry a `name` column (see population.add_hero_names) for the hero labels.
Colors and fonts come from the active theme (theme.py), never from constants here.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402
from scipy.stats import norm  # noqa: E402

from winlock import theme  # noqa: E402
from winlock.portraits import load_portraits  # noqa: E402


def _save(fig: plt.Figure, path: Path, dpi: int) -> Path:
    theme.finish(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A framed theme lays itself out (constrained layout); cropping would cut the frame.
    fig.savefig(path, dpi=dpi, bbox_inches=None if theme.active().trim else "tight")
    plt.close(fig)
    return path


def _portrait_axis(ax, hero_ids, y, portraits_dir: Path | None, t: theme.Theme) -> None:
    """Small round portraits just left of the axis, one per row; tick labels move further left.

    Heroes without a cached portrait just keep their name.
    """
    images = load_portraits(portraits_dir, hero_ids)
    if not images:
        return
    size_pt = t.row_height * 72 * 0.8  # 80% of the row height, in points
    for hero_id, yi in zip(hero_ids, y, strict=True):
        img = images.get(int(hero_id))
        if img is None:
            continue
        box = AnnotationBbox(
            OffsetImage(img, zoom=size_pt / img.shape[0]),
            (0, yi),
            xycoords=("axes fraction", "data"),
            xybox=(-size_pt / 2 - 3, 0),
            boxcoords="offset points",
            frameon=True,
            bboxprops={
                "boxstyle": "circle,pad=0.0",
                "edgecolor": t.trim or t.grid,
                "facecolor": t.surface,
                "linewidth": 0.8,
            },
            annotation_clip=False,
        )
        ax.add_artist(box)
    ax.tick_params(axis="y", length=0, pad=size_pt + 8)


def winrate_ci_chart(
    df: pd.DataFrame,
    path: Path,
    title: str,
    alpha: float = 0.05,
    dpi: int = 150,
    portraits_dir: Path | None = None,
) -> Path:
    """Heroes sorted by win rate, Wilson CI as a line, the 50% line highlighted.

    Colored = different from 50% after Holm correction over all heroes (`p_vs_half_holm`, see
    stats.binomial_vs_half), `high` color above 50% and `low` below (the same color in the
    clean theme); gray = can't tell from 50%. Without that column, falls back to "CI excludes
    50%" (uncorrected). With `portraits_dir`, cached hero portraits sit next to the names.
    """
    d = df.sort_values("win_rate").reset_index(drop=True)
    if "p_vs_half_holm" in d:
        differs = d["p_vs_half_holm"] < alpha
        tag, none_label = "(Holm)", "not distinguishable from 50% (Holm)"
    else:
        differs = (d["ci_low"] > 0.5) | (d["ci_high"] < 0.5)
        tag, none_label = "(95% CI)", "95% CI includes 50%"
    above = d["win_rate"] > 0.5
    y = np.arange(len(d))
    with theme.style() as t:
        if t.high == t.low:  # one color for "differs": one legend entry
            groups = [(differs, t.high, f"differs from 50% {tag}")]
        else:
            groups = [
                (differs & above, t.high, f"above 50% {tag}"),
                (differs & ~above, t.low, f"below 50% {tag}"),
            ]
        groups.append((~differs, t.neutral, none_label))
        fig, ax = plt.subplots(figsize=(7, t.row_height * len(d) + 1.5))
        ax.grid(axis="y", visible=False)
        ax.axvline(0.5, color=t.text, linewidth=1.2, zorder=1)
        for mask, color, label in groups:
            if not mask.any():
                continue
            ax.hlines(y[mask], d["ci_low"][mask], d["ci_high"][mask], color=color, linewidth=2)
            ax.scatter(
                d["win_rate"][mask],
                y[mask],
                color=color,
                s=26,
                zorder=3,
                edgecolor=t.surface,
                linewidth=1,
                label=label,
            )
        ax.set_yticks(y, d["name"])
        _portrait_axis(ax, d["hero_id"], y, portraits_dir, t)
        ax.set_ylim(-0.8, len(d) - 0.2)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("Win rate (dot) with 95% Wilson CI (line)")
        ax.set_title(title)
        # Below the axis: inside, it sat on the 50% line next to the lowest heroes. The anchor
        # is 34 points under the axis (clear of ticks + x label), whatever the chart's height;
        # an axes fraction like -0.07 would grow with the number of heroes.
        below = offset_copy(ax.transAxes, fig=fig, y=-34, units="points")
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, 0), bbox_transform=below, ncol=3, fontsize=8
        )
        return _save(fig, path, dpi)


def games_per_hero_chart(df: pd.DataFrame, path: Path, title: str, dpi: int = 150) -> Path:
    """Sample size per hero: fewer games means a wider CI."""
    d = df.sort_values("games").reset_index(drop=True)
    y = np.arange(len(d))
    with theme.style() as t:
        fig, ax = plt.subplots(figsize=(7, t.row_height * len(d) + 1.5))
        ax.grid(axis="y", visible=False)
        ax.barh(y, d["games"], height=0.7, color=t.accent)
        ax.set_yticks(y, d["name"])
        ax.set_ylim(-0.8, len(d) - 0.2)
        ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_xlabel("Games (player rows)")
        ax.set_title(title)
        return _save(fig, path, dpi)


def ci_width_vs_games_chart(
    df: pd.DataFrame, path: Path, title: str, alpha: float = 0.05, dpi: int = 150
) -> Path:
    """CI width against games, with the p = 0.5 curve 2·z·sqrt(0.25/n) for reference."""
    z = norm.ppf(1 - alpha / 2)
    n = np.linspace(df["games"].min() * 0.9, df["games"].max() * 1.05, 200)
    with theme.style() as t:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(
            n,
            100 * 2 * z * np.sqrt(0.25 / n),
            color=t.text_secondary,
            linewidth=2,
            label="2·z·√(0.25/n)",
        )
        ax.scatter(
            df["games"],
            100 * (df["ci_high"] - df["ci_low"]),
            color=t.accent,
            s=28,
            zorder=3,
            edgecolor=t.surface,
            linewidth=1.5,
            label="heroes",
        )
        ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_xlabel("Games (player rows)")
        ax.set_ylabel("95% CI width (percentage points)")
        ax.set_title(title)
        ax.legend(loc="upper right")
        return _save(fig, path, dpi)


def _label_points(ax, x, y, names) -> None:
    """Direct labels for a few selected points, in secondary ink."""
    t = theme.active()
    for xi, yi, name in zip(x, y, names, strict=True):
        ax.annotate(
            name,
            (xi, yi),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=8,
            color=t.text_secondary,
        )


def _percent_axis(axis, step: float = 0.02) -> None:
    axis.set_major_locator(matplotlib.ticker.MultipleLocator(step))
    axis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def train_vs_test_chart(
    ev: pd.DataFrame, path: Path, title: str, n_labels: int = 6, dpi: int = 150
) -> Path:
    """Each hero's train win rate (x) against its test win rate (y).

    On the y = x line the rate held exactly. Accent = still significant after Holm correction.
    The heroes with the largest |z| are labelled.
    """
    lo = min(ev["rate_train"].min(), ev["rate_test"].min()) - 0.01
    hi = max(ev["rate_train"].max(), ev["rate_test"].max()) + 0.01
    moved = ev["p_holm"] < 0.05
    with theme.style() as t:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.plot([lo, hi], [lo, hi], color=t.text_secondary, linewidth=1.2, label="test = train")
        ax.axhline(0.5, color=t.grid, linewidth=1.5, zorder=0)
        ax.axvline(0.5, color=t.grid, linewidth=1.5, zorder=0)
        for mask, color, label in [
            (~moved, t.neutral, "change not significant (Holm)"),
            (moved, t.accent, "change significant (Holm)"),
        ]:
            if not mask.any():  # no empty legend entries
                continue
            ax.scatter(
                ev["rate_train"][mask],
                ev["rate_test"][mask],
                color=color,
                s=36,
                zorder=3,
                edgecolor=t.surface,
                linewidth=1.5,
                label=label,
            )
        top = ev.reindex(ev["z"].abs().sort_values(ascending=False).index).head(n_labels)
        _label_points(ax, top["rate_train"], top["rate_test"], top["name"])
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        _percent_axis(ax.xaxis)
        _percent_axis(ax.yaxis)
        ax.set_xlabel("Win rate in train (00:00–13:35 UTC)")
        ax.set_ylabel("Win rate in test (13:35–22:00 UTC)")
        ax.set_title(title)
        ax.legend(loc="upper left")
        return _save(fig, path, dpi)


def z_distribution_chart(zs: dict[str, pd.Series], path: Path, title: str, dpi: int = 150) -> Path:
    """Histograms (as outlines) of z-scores between two samples, against the N(0, 1) curve
    they'd follow by pure chance. One outline per comparison (at most two).
    """
    grid = np.linspace(-4.5, 4.5, 300)
    bound = max(4.5, max(float(np.ceil(np.abs(z).max())) for z in zs.values()))
    with theme.style() as t:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for (name, z), color in zip(zs.items(), [t.accent, t.accent_2][: len(zs)], strict=True):
            ax.hist(
                z,
                bins=np.arange(-bound, bound + 0.5, 0.5),
                density=True,
                histtype="step",
                color=color,
                linewidth=2.5,
                label=f"{name} (SD of z = {np.std(z, ddof=1):.2f})",
            )
        ax.plot(
            grid,
            norm.pdf(grid),
            color=t.text,
            linewidth=1.5,
            linestyle="--",
            label="N(0, 1): pure chance",
        )
        ax.set_xlabel("z = (second rate − first rate) / combined SE, per hero")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
        return _save(fig, path, dpi)


def coverage_simulation_chart(
    simulated: np.ndarray, observed: float, path: Path, title: str, dpi: int = 150
) -> Path:
    """Distribution of coverage in simulated "nothing changed" worlds, with the observed value."""
    with theme.style() as t:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bins = np.arange(simulated.min() - 0.0125, simulated.max() + 0.025, 1 / 40)
        ax.hist(
            simulated,
            bins=bins,
            color=t.neutral,
            edgecolor=t.surface,
            linewidth=1,
            label="simulated: nothing changed",
        )
        ax.axvline(observed, color=t.accent, linewidth=2.5, label=f"observed: {observed:.0%}")
        ax.axvline(0.95, color=t.text, linewidth=1.2, linestyle="--", label="nominal 95%")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("Share of heroes whose test rate falls inside their train CI")
        ax.set_ylabel("Simulations")
        ax.set_title(title)
        ax.legend(loc="upper left")
        return _save(fig, path, dpi)


def model_accuracy_chart(metrics: pd.DataFrame, path: Path, title: str, dpi: int = 150) -> Path:
    """Test accuracy per model as a dot with its 95% Wilson CI, against the 50% line."""
    y = np.arange(len(metrics))
    with theme.style() as t:
        fig, ax = plt.subplots(figsize=(7, 0.6 * len(metrics) + 1.5))
        ax.grid(axis="y", visible=False)
        ax.axvline(0.5, color=t.text, linewidth=1.2, zorder=1)
        ax.hlines(y, metrics["acc_ci_low"], metrics["acc_ci_high"], color=t.accent, linewidth=2)
        ax.scatter(metrics["accuracy"], y, color=t.accent, s=36, zorder=3)
        for yi, acc in zip(y, metrics["accuracy"], strict=True):
            ax.annotate(
                f"{acc:.1%}",
                (acc, yi),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=t.text_secondary,
            )
        ax.set_yticks(y, metrics["model"])
        ax.set_ylim(-0.7, len(metrics) - 0.3)
        _percent_axis(ax.xaxis, step=0.01)
        ax.set_xlabel("Test accuracy (dot) with 95% Wilson CI (line)")
        ax.set_title(title)
        return _save(fig, path, dpi)


def calibration_chart(
    preds: dict[str, tuple[np.ndarray, np.ndarray]],
    path: Path,
    title: str,
    n_bins: int = 10,
    dpi: int = 150,
) -> Path:
    """Predicted P(Team0 wins) vs how often Team0 actually won, in deciles of the prediction."""
    with theme.style() as t:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.plot([0, 1], [0, 1], color=t.text_secondary, linewidth=1.2, label="perfect calibration")
        for (name, (y, p)), color in zip(
            preds.items(), [t.accent, t.accent_2][: len(preds)], strict=True
        ):
            d = pd.DataFrame({"y": y, "p": p})
            d["bin"] = pd.qcut(d["p"], n_bins, labels=False, duplicates="drop")
            g = d.groupby("bin").mean()
            ax.plot(
                g["p"],
                g["y"],
                color=color,
                linewidth=2,
                marker="o",
                markersize=6,
                markeredgecolor=t.surface,
                label=name,
            )
        ax.set_xlim(0.2, 0.8)
        ax.set_ylim(0.2, 0.8)
        ax.set_aspect("equal")
        _percent_axis(ax.xaxis, step=0.1)
        _percent_axis(ax.yaxis, step=0.1)
        ax.set_xlabel("Predicted P(Team0 wins), decile average")
        ax.set_ylabel("Observed Team0 win share")
        ax.set_title(title)
        ax.legend(loc="upper left")
        return _save(fig, path, dpi)


def coefficients_chart(
    coef: pd.DataFrame, path: Path, title: str, n_outliers: int = 3, dpi: int = 150
) -> Path:
    """Logistic regression coefficient per hero against the hero's train win rate.

    Labels: the highest and lowest coefficient, plus the heroes furthest from the linear trend
    (where the model disagrees most with the plain win rate).
    """
    with theme.style() as t:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.axhline(0, color=t.grid, linewidth=1.5, zorder=0)
        ax.axvline(0.5, color=t.grid, linewidth=1.5, zorder=0)
        ax.scatter(
            coef["rate_train"],
            coef["coef"],
            color=t.accent,
            s=36,
            zorder=3,
            edgecolor=t.surface,
            linewidth=1.5,
        )
        slope, intercept = np.polyfit(coef["rate_train"], coef["coef"], 1)
        residual = (coef["coef"] - (slope * coef["rate_train"] + intercept)).abs()
        picks = {coef["coef"].idxmax(), coef["coef"].idxmin()}
        picks |= set(residual.drop(list(picks)).nlargest(n_outliers).index)
        top = coef.loc[sorted(picks)]
        _label_points(ax, top["rate_train"], top["coef"], top["name"])
        _percent_axis(ax.xaxis)
        ax.set_xlabel("Train win rate")
        ax.set_ylabel("Logistic regression coefficient (log-odds)")
        ax.set_title(title)
        return _save(fig, path, dpi)


def pair_matches_chart(pairs: pd.DataFrame, path: Path, title: str, dpi: int = 150) -> Path:
    """How many matches each teammate pair played together (log scale): the sample-size problem."""
    bins = np.logspace(0, np.log10(pairs["matches"].max() * 1.1), 30)
    with theme.style() as t:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(pairs["matches"], bins=bins, color=t.accent, edgecolor=t.surface, linewidth=1)
        ax.set_xscale("log")
        ax.set_xlim(left=10)
        for n in (100, 1000):
            ax.axvline(n, color=t.text_secondary, linewidth=1, linestyle="--")
            ax.annotate(
                f"95% CI ≈ ±{100 * 1.96 * np.sqrt(0.25 / n):.0f} pp",
                (n, ax.get_ylim()[1] * 0.92),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=8,
                color=t.text_secondary,
            )
        ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_xlabel("Matches the pair played together on the same team (train)")
        ax.set_ylabel("Pairs")
        ax.set_title(title)
        return _save(fig, path, dpi)


def synergy_volcano_chart(
    pairs: pd.DataFrame,
    path: Path,
    title: str,
    alpha: float = 0.05,
    n_labels: int = 4,
    dpi: int = 150,
) -> Path:
    """Excess win rate of each pair (x) against its evidence, −log10(p) (y).

    The dashed line is the bar the strongest pair must clear after Holm (alpha / number of
    pairs). Accent = significant after Holm.
    """
    sig = pairs["p_holm"] < alpha
    y = -np.log10(pairs["p"].clip(lower=1e-300))
    with theme.style() as t:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.axvline(0, color=t.grid, linewidth=1.5, zorder=0)
        ax.axhline(
            -np.log10(alpha),
            color=t.text_secondary,
            linewidth=1,
            linestyle=":",
            label=f"p = {alpha} (no correction)",
        )
        ax.axhline(
            -np.log10(alpha / len(pairs)),
            color=t.text,
            linewidth=1.2,
            linestyle="--",
            label=f"Holm bar for {len(pairs):,} pairs",
        )
        for mask, color, label in [
            (~sig, t.neutral, "not significant (Holm)"),
            (sig, t.accent, "significant (Holm)"),
        ]:
            if not mask.any():
                continue
            ax.scatter(
                100 * pairs["excess"][mask],
                y[mask],
                color=color,
                s=14,
                alpha=0.8,
                zorder=3,
                linewidth=0,
                label=label,
            )
        top = pairs.loc[y.sort_values(ascending=False).index[:n_labels]]
        for k, (xi, yi, name) in enumerate(
            zip(100 * top["excess"], y[top.index], top["pair"], strict=True)
        ):
            # alternate above/below so neighbouring labels don't collide
            ax.annotate(
                name,
                (xi, yi),
                xytext=(5, 4 if k % 2 == 0 else -11),
                textcoords="offset points",
                fontsize=8,
                color=t.text_secondary,
            )
        ax.set_ylim(0, -np.log10(alpha / len(pairs)) * 1.12)
        ax.set_xlabel("Excess win rate together vs additive prediction (pp)")
        ax.set_ylabel("Evidence, −log10(p)")
        ax.set_title(title)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
        return _save(fig, path, dpi)


def synergy_replication_chart(pairs: pd.DataFrame, path: Path, title: str, dpi: int = 150) -> Path:
    """Each pair's synergy z on train (x) vs on test (y). Real synergy lines up; noise doesn't."""
    corr = np.corrcoef(pairs["z_train"], pairs["z_test"])[0, 1]
    lim = float(np.ceil(max(pairs["z_train"].abs().max(), pairs["z_test"].abs().max())))
    with theme.style() as t:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.axhline(0, color=t.grid, linewidth=1.5, zorder=0)
        ax.axvline(0, color=t.grid, linewidth=1.5, zorder=0)
        ax.plot(
            [-lim, lim],
            [-lim, lim],
            color=t.text_secondary,
            linewidth=1.2,
            label="same synergy in both",
        )
        ax.scatter(
            pairs["z_train"],
            pairs["z_test"],
            color=t.accent,
            s=14,
            alpha=0.6,
            linewidth=0,
            zorder=3,
            label=f"pairs (correlation {corr:.2f})",
        )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("Synergy z on train")
        ax.set_ylabel("Synergy z on test")
        ax.set_title(title)
        ax.legend(loc="upper left")
        return _save(fig, path, dpi)
