"""Chart themes: one palette + matplotlib style per look, chosen in config ([figures] theme).

figures.py never hard-codes colors: it asks `active()` for the current Theme. The CLI calls
`use(name)` once with the configured name. Two looks:

- "clean": light, one blue accent (the original reference style).
- "deadlock": dark smoke, cream text, brass art-deco trim, and the game's own team colors
  (Amber = team1_color, Sapphire = team2_color, from the deadlock-api /v1/assets/colors list).

Fonts are open-license (OFL) look-alikes bundled in assets/fonts/; the game's own fonts
(Radiance, Reaver) are commercial and can't be redistributed.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, RegularPolygon

FONTS_DIR = Path(__file__).parent / "assets" / "fonts"


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str  # background
    text: str  # titles, the 50% line
    text_secondary: str  # labels, ticks, annotations
    grid: str
    accent: str  # the single-series color
    accent_2: str  # second series
    high: str  # "differs from 50%, above"
    low: str  # "differs from 50%, below"
    neutral: str  # "can't tell from 50%"
    trim: str | None = None  # art-deco frame color; None = no frame
    title_font: str | None = None  # None = same as body text
    uppercase_titles: bool = False
    row_height: float = 0.22  # inches per hero row in the per-hero charts
    rc: dict = field(default_factory=dict)


def _rc(t_surface, t_text, t_secondary, t_grid, extra=None) -> dict:
    """The matplotlib settings every theme shares, filled with that theme's colors."""
    rc = {
        "figure.facecolor": t_surface,
        "axes.facecolor": t_surface,
        "savefig.facecolor": t_surface,
        "axes.edgecolor": t_grid,
        "axes.labelcolor": t_secondary,
        "axes.titlecolor": t_text,
        "axes.titlesize": 13,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": t_grid,
        "grid.linewidth": 0.8,
        "xtick.color": t_secondary,
        "ytick.color": t_secondary,
        "text.color": t_text,
        "legend.labelcolor": t_secondary,
        "font.size": 9,
        "legend.frameon": False,
    }
    return rc | (extra or {})


CLEAN = Theme(
    name="clean",
    surface="#fcfcfb",
    text="#0b0b0b",
    text_secondary="#52514e",
    grid="#e4e3de",
    accent="#2a78d6",
    accent_2="#eb6834",
    high="#2a78d6",
    low="#2a78d6",
    neutral="#a3a29c",
    rc=_rc("#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"),
)

# Deadlock: in-game base_text #ffefd7, golden_light #c6a269, golden_dark #92703c,
# team1_color (Amber) #d4860b, team2_color (Sapphire) #4d75c3. Smoke tones are ours.
_SMOKE, _CREAM, _BRASS_LIGHT, _BRASS_DARK = "#15120f", "#ffefd7", "#c6a269", "#92703c"
DEADLOCK = Theme(
    name="deadlock",
    surface=_SMOKE,
    text=_CREAM,
    text_secondary="#b8aa90",
    grid="#2c2620",
    accent="#d4860b",
    accent_2="#4d75c3",
    high="#d4860b",
    low="#6f93dc",  # Sapphire lifted a step: #4d75c3 is too dark on smoke to read as a thin line
    neutral="#7a705e",
    trim=_BRASS_DARK,
    title_font="Cinzel Decorative",
    uppercase_titles=True,
    row_height=0.30,  # room for the portraits
    rc=_rc(
        _SMOKE,
        _CREAM,
        "#b8aa90",
        "#2c2620",
        {
            # Glyphs Josefin lacks (→, the − minus sign) fall back to DejaVu, one by one.
            "font.family": ["Josefin Sans", "DejaVu Sans"],
            # ...but the fallback − is wider than Josefin's digits ("- 2"): use its own hyphen.
            "axes.unicode_minus": False,
            "font.size": 9.5,
            "axes.edgecolor": _BRASS_DARK,
            "axes.titlecolor": _BRASS_LIGHT,
            "axes.titlesize": 14,
            "axes.titlepad": 14,
            "grid.linestyle": (0, (1, 3)),  # dotted, like engraved guide lines
            # Lay out inside the figure (not crop afterwards) and keep clear of the frame.
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.w_pad": 0.3,
            "figure.constrained_layout.h_pad": 0.3,
        },
    ),
)

THEMES = {t.name: t for t in (CLEAN, DEADLOCK)}
_active: Theme = DEADLOCK


def register_fonts() -> None:
    """Make the bundled .ttf files known to matplotlib (safe to call more than once)."""
    for path in FONTS_DIR.glob("*.ttf"):
        font_manager.fontManager.addfont(str(path))


def use(name: str) -> Theme:
    """Pick the theme every chart will use from now on."""
    global _active
    if name not in THEMES:
        raise ValueError(f"unknown theme {name!r}; choose one of {sorted(THEMES)}")
    _active = THEMES[name]
    register_fonts()
    return _active


def active() -> Theme:
    return _active


@contextmanager
def style() -> Iterator[Theme]:
    """`with theme.style() as t:` applies the active theme's rc settings inside the block."""
    with plt.rc_context(_active.rc):
        yield _active


def finish(fig: plt.Figure) -> None:
    """Theme touches applied right before saving: title font/case and the art-deco frame."""
    t = _active
    for ax in fig.axes:
        # Each axes has three title slots; ours lives in "left" (axes.titlelocation).
        for loc in ("left", "center", "right"):
            text = ax.get_title(loc)
            if not text:
                continue
            ax.set_title(
                text.upper() if t.uppercase_titles else text,
                loc=loc,
                fontfamily=[t.title_font, "DejaVu Sans"] if t.title_font else None,
            )
    if t.trim:
        _deco_frame(fig, t.trim)


def _deco_frame(fig: plt.Figure, color: str) -> None:
    """Double brass border with small diamonds at the corners, in figure coordinates (0–1)."""
    w, h = fig.get_size_inches()
    m = 0.08  # outer margin, inches
    gap = 0.05  # space between the two lines, inches
    for inset, lw in ((m, 1.4), (m + gap, 0.6)):
        fig.add_artist(
            Rectangle(
                (inset / w, inset / h),
                1 - 2 * inset / w,
                1 - 2 * inset / h,
                transform=fig.transFigure,
                fill=False,
                edgecolor=color,
                linewidth=lw,
                zorder=-1,
            )
        )
    r = 0.07  # diamond radius, inches; drawn in inch units so it stays square
    for x, y in ((m, m), (w - m, m), (m, h - m), (w - m, h - m)):
        fig.add_artist(
            RegularPolygon(
                (x, y),
                numVertices=4,
                radius=r,
                transform=fig.dpi_scale_trans,
                facecolor=color,
                edgecolor="none",
            )
        )
