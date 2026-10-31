"""Theme switching and fonts."""

import pytest
from matplotlib import font_manager

from winlock import theme


@pytest.fixture(autouse=True)
def restore_theme():
    before = theme.active().name
    yield
    theme.use(before)


def test_use_switches_active_theme():
    assert theme.use("clean") is theme.CLEAN
    assert theme.active() is theme.CLEAN
    theme.use("deadlock")
    assert theme.active() is theme.DEADLOCK


def test_unknown_theme_is_an_error():
    with pytest.raises(ValueError, match="unknown theme"):
        theme.use("neon")


def test_bundled_fonts_are_registered():
    theme.register_fonts()
    names = {f.name for f in font_manager.fontManager.ttflist}
    assert {"Cinzel Decorative", "Josefin Sans"} <= names


def test_deadlock_uses_the_game_team_colors():
    # team1_color (Amber) and team2_color (Sapphire) from the deadlock-api colors list
    assert theme.DEADLOCK.accent == "#d4860b"
    assert theme.DEADLOCK.accent_2 == "#4d75c3"
