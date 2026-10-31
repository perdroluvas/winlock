"""Charts for the post. Each function takes an aggregate table and writes one PNG.

Tables must carry a `name` column (see population.add_hero_names) for the hero labels.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

# Reference palette (light mode): one accent for the signal, gray for "can't tell from 50%".
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3de"
ACCENT = "#2a78d6"
ACCENT_2 = "#eb6834"  # second categorical slot (orange)
NEUTRAL = "#a3a29c"

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT_PRIMARY,
    "axes.titlesize": 13,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "font.size": 9,
    "legend.frameon": False,
}


def _save(fig: plt.Figure, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def winrate_ci_chart(
    df: pd.DataFrame, path: Path, title: str, alpha: float = 0.05, dpi: int = 150
) -> Path:
    """Heroes sorted by win rate, Wilson CI as a line, the 50% line highlighted.

    Accent = different from 50% after Holm correction over all heroes (`p_vs_half_holm`, see
    stats.binomial_vs_half); gray = can't tell from 50%. Without that column, falls back to
    "CI excludes 50%" (uncorrected).
    """
    d = df.sort_values("win_rate").reset_index(drop=True)
    if "p_vs_half_holm" in d:
        differs = d["p_vs_half_holm"] < alpha
        labels = ("differs from 50% (Holm)", "not distinguishable from 50% (Holm)")
    else:
        differs = (d["ci_low"] > 0.5) | (d["ci_high"] < 0.5)
        labels = ("95% CI excludes 50%", "95% CI includes 50%")
    y = np.arange(len(d))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 0.22 * len(d) + 1.5))
        ax.grid(axis="y", visible=False)
        ax.axvline(0.5, color=TEXT_PRIMARY, linewidth=1.2, zorder=1)
        for mask, color, label in [(differs, ACCENT, labels[0]), (~differs, NEUTRAL, labels[1])]:
            if not mask.any():
                continue
            ax.hlines(y[mask], d["ci_low"][mask], d["ci_high"][mask], color=color, linewidth=2)
            ax.scatter(d["win_rate"][mask], y[mask], color=color, s=24, zorder=3, label=label)
        ax.set_yticks(y, d["name"])
        ax.set_ylim(-0.8, len(d) - 0.2)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("Win rate (dot) with 95% Wilson CI (line)")
        ax.set_title(title)
        ax.legend(loc="lower right")
        return _save(fig, path, dpi)


def games_per_hero_chart(df: pd.DataFrame, path: Path, title: str, dpi: int = 150) -> Path:
    """Sample size per hero: fewer games means a wider CI."""
    d = df.sort_values("games").reset_index(drop=True)
    y = np.arange(len(d))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 0.22 * len(d) + 1.5))
        ax.grid(axis="y", visible=False)
        ax.barh(y, d["games"], height=0.7, color=ACCENT)
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
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(
            n,
            100 * 2 * z * np.sqrt(0.25 / n),
            color=TEXT_SECONDARY,
            linewidth=2,
            label="2·z·√(0.25/n)",
        )
        ax.scatter(
            df["games"],
            100 * (df["ci_high"] - df["ci_low"]),
            color=ACCENT,
            s=28,
            zorder=3,
            edgecolor=SURFACE,
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
    for xi, yi, name in zip(x, y, names, strict=True):
        ax.annotate(
            name,
            (xi, yi),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=8,
            color=TEXT_SECONDARY,
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
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.plot([lo, hi], [lo, hi], color=TEXT_SECONDARY, linewidth=1.2, label="test = train")
        ax.axhline(0.5, color=GRID, linewidth=1.5, zorder=0)
        ax.axvline(0.5, color=GRID, linewidth=1.5, zorder=0)
        for mask, color, label in [
            (~moved, NEUTRAL, "change not significant (Holm)"),
            (moved, ACCENT, "change significant (Holm)"),
        ]:
            if not mask.any():  # no empty legend entries
                continue
            ax.scatter(
                ev["rate_train"][mask],
                ev["rate_test"][mask],
                color=color,
                s=36,
                zorder=3,
                edgecolor=SURFACE,
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
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for (name, z), color in zip(zs.items(), [ACCENT, ACCENT_2][: len(zs)], strict=True):
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
            color=TEXT_PRIMARY,
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
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bins = np.arange(simulated.min() - 0.0125, simulated.max() + 0.025, 1 / 40)
        ax.hist(
            simulated,
            bins=bins,
            color=NEUTRAL,
            edgecolor=SURFACE,
            linewidth=1,
            label="simulated: nothing changed",
        )
        ax.axvline(observed, color=ACCENT, linewidth=2.5, label=f"observed: {observed:.0%}")
        ax.axvline(0.95, color=TEXT_PRIMARY, linewidth=1.2, linestyle="--", label="nominal 95%")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("Share of heroes whose test rate falls inside their train CI")
        ax.set_ylabel("Simulations")
        ax.set_title(title)
        ax.legend(loc="upper left")
        return _save(fig, path, dpi)


def model_accuracy_chart(metrics: pd.DataFrame, path: Path, title: str, dpi: int = 150) -> Path:
    """Test accuracy per model as a dot with its 95% Wilson CI, against the 50% line."""
    y = np.arange(len(metrics))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 0.6 * len(metrics) + 1.5))
        ax.grid(axis="y", visible=False)
        ax.axvline(0.5, color=TEXT_PRIMARY, linewidth=1.2, zorder=1)
        ax.hlines(y, metrics["acc_ci_low"], metrics["acc_ci_high"], color=ACCENT, linewidth=2)
        ax.scatter(metrics["accuracy"], y, color=ACCENT, s=36, zorder=3)
        for yi, acc in zip(y, metrics["accuracy"], strict=True):
            ax.annotate(
                f"{acc:.1%}",
                (acc, yi),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=TEXT_SECONDARY,
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
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linewidth=1.2, label="perfect calibration")
        for (name, (y, p)), color in zip(
            preds.items(), [ACCENT, ACCENT_2][: len(preds)], strict=True
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
                markeredgecolor=SURFACE,
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
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.axhline(0, color=GRID, linewidth=1.5, zorder=0)
        ax.axvline(0.5, color=GRID, linewidth=1.5, zorder=0)
        ax.scatter(
            coef["rate_train"],
            coef["coef"],
            color=ACCENT,
            s=36,
            zorder=3,
            edgecolor=SURFACE,
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
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(pairs["matches"], bins=bins, color=ACCENT, edgecolor=SURFACE, linewidth=1)
        ax.set_xscale("log")
        ax.set_xlim(left=10)
        for n in (100, 1000):
            ax.axvline(n, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
            ax.annotate(
                f"95% CI ≈ ±{100 * 1.96 * np.sqrt(0.25 / n):.0f} pp",
                (n, ax.get_ylim()[1] * 0.92),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=8,
                color=TEXT_SECONDARY,
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
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.axvline(0, color=GRID, linewidth=1.5, zorder=0)
        ax.axhline(
            -np.log10(alpha),
            color=TEXT_SECONDARY,
            linewidth=1,
            linestyle=":",
            label=f"p = {alpha} (no correction)",
        )
        ax.axhline(
            -np.log10(alpha / len(pairs)),
            color=TEXT_PRIMARY,
            linewidth=1.2,
            linestyle="--",
            label=f"Holm bar for {len(pairs):,} pairs",
        )
        for mask, color, label in [
            (~sig, NEUTRAL, "not significant (Holm)"),
            (sig, ACCENT, "significant (Holm)"),
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
                color=TEXT_SECONDARY,
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
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.axhline(0, color=GRID, linewidth=1.5, zorder=0)
        ax.axvline(0, color=GRID, linewidth=1.5, zorder=0)
        ax.plot(
            [-lim, lim],
            [-lim, lim],
            color=TEXT_SECONDARY,
            linewidth=1.2,
            label="same synergy in both",
        )
        ax.scatter(
            pairs["z_train"],
            pairs["z_test"],
            color=ACCENT,
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
