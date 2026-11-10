"""Charts for the article."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

from winlock import theme  # noqa: E402
from winlock.portraits import load_portraits  # noqa: E402


def _save(fig: plt.Figure, path: Path, dpi: int) -> Path:
    theme.finish(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=None if theme.active().trim else "tight")
    plt.close(fig)
    return path


def _portrait_axis(ax, hero_ids, y, portraits_dir: Path | None, t: theme.Theme) -> None:
    images = load_portraits(portraits_dir, hero_ids)
    if not images:
        return
    size_pt = t.row_height * 72 * 0.8
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
        if t.high == t.low:
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
        below = offset_copy(ax.transAxes, fig=fig, y=-34, units="points")
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, 0), bbox_transform=below, ncol=3, fontsize=8
        )
        return _save(fig, path, dpi)


def _label_points(ax, x, y, names) -> None:
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


def before_after_chart(
    ev: pd.DataFrame,
    path: Path,
    title: str,
    n_labels: int = 6,
    dpi: int = 150,
    names: tuple[str, str] = ("before", "after"),
    axis_labels: tuple[str, str] = ("Win rate before (24 h)", "Win rate after (24 h)"),
) -> Path:
    x, y = (ev[f"rate_{n}"] for n in names)
    lo = min(x.min(), y.min()) - 0.01
    hi = max(x.max(), y.max()) + 0.01
    moved = ev["p_holm"] < 0.05
    with theme.style() as t:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.plot(
            [lo, hi],
            [lo, hi],
            color=t.text_secondary,
            linewidth=1.2,
            label=f"{names[1]} = {names[0]}",
        )
        ax.axhline(0.5, color=t.grid, linewidth=1.5, zorder=0)
        ax.axvline(0.5, color=t.grid, linewidth=1.5, zorder=0)
        for mask, color, label in [
            (~moved, t.neutral, "change not significant (Holm)"),
            (moved, t.accent, "change significant (Holm)"),
        ]:
            if not mask.any():
                continue
            ax.scatter(
                x[mask],
                y[mask],
                color=color,
                s=36,
                zorder=3,
                edgecolor=t.surface,
                linewidth=1.5,
                label=label,
            )
        top = ev.reindex(ev["z"].abs().sort_values(ascending=False).index).head(n_labels)
        _label_points(ax, top[f"rate_{names[0]}"], top[f"rate_{names[1]}"], top["name"])
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        _percent_axis(ax.xaxis)
        _percent_axis(ax.yaxis)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_title(title)
        ax.legend(loc="upper left")
        return _save(fig, path, dpi)


def model_accuracy_chart(metrics: pd.DataFrame, path: Path, title: str, dpi: int = 150) -> Path:
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


def patch_dip_chart(
    starts: pd.DataFrame,
    path: Path,
    title: str,
    update_at: pd.Timestamp,
    buffer: tuple[pd.Timestamp, pd.Timestamp],
    dpi: int = 150,
) -> Path:
    with theme.style() as t:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.axvspan(*buffer, color=t.grid, alpha=0.6, linewidth=0, label="buffer (excluded)")
        ax.plot(starts["bin_start"], starts["matches"], color=t.accent, linewidth=2)
        ax.axvline(update_at, color=t.text, linewidth=1.2, linestyle="--", label="update published")
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M", tz=update_at.tz))
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        minutes = int(
            (starts["bin_start"].iloc[1] - starts["bin_start"].iloc[0]).total_seconds() // 60
        )
        ax.set_xlabel(f"Start time, {update_at:%Y-%m-%d} (UTC)")
        ax.set_ylabel(f"Matches started per {minutes} min")
        ax.set_title(title)
        ax.legend(loc="lower left")
        return _save(fig, path, dpi)
