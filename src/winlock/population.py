"""The analysis population and the named time windows inside it."""

from typing import Any

import duckdb
import pandas as pd

# Only the columns the analysis needs; account_id never enters the population table.
POPULATION_COLUMNS = ["match_id", "hero_id", "team", "won", "start_time"]

ROW_FILTERS = [
    (
        "window",
        "start_time >= CAST($start AS TIMESTAMPTZ) AND start_time < CAST($end AS TIMESTAMPTZ)",
    ),
    ("game_mode", "game_mode = $game_mode"),
    ("match_mode", "list_contains($match_mode, match_mode)"),
    ("match_outcome", "match_outcome = $match_outcome"),
]

Window = tuple[str, str]


def window(cfg: dict[str, Any], name: str) -> Window:
    """(start, end) of a named window in cfg["windows"]."""
    raise NotImplementedError


def population_span(cfg: dict[str, Any]) -> Window:
    """(earliest start, latest end) over all named windows."""
    raise NotImplementedError


def _filter_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """Query parameters for ROW_FILTERS: start and end (the population span) plus one
    entry per population filter (game_mode, match_mode, match_outcome).
    """
    raise NotImplementedError


def create_population(con: duckdb.DuckDBPyConnection, cfg: dict[str, Any]) -> None:
    """Complete matches only (12 rows, 6 winners), dropped whole otherwise.

    Creates the table `population` (POPULATION_COLUMNS) from the view `match_player`: rows
    that pass ROW_FILTERS and carry none of cfg["population"]["exclude_flags"], in matches
    left with exactly players_per_match rows and wins_per_match winners.
    """
    raise NotImplementedError


def population_report(con: duckdb.DuckDBPyConnection, cfg: dict[str, Any]) -> pd.DataFrame:
    """How many rows and matches survive each filter, cumulatively.

    One row per step: `file`, then each ROW_FILTERS key in order, then `complete_match`.
    Columns: step, rows, matches, pct_matches_kept (vs the previous step, 1 decimal).
    """
    raise NotImplementedError


def in_window(win: Window) -> str:
    """SQL condition: start_time inside [start, end)."""
    raise NotImplementedError


def window_sizes(con: duckdb.DuckDBPyConnection, cfg: dict[str, Any]) -> pd.DataFrame:
    """Per named window: window, matches, first_start, last_start (as '%m-%d %H:%M')."""
    raise NotImplementedError


def hero_winrates(con: duckdb.DuckDBPyConnection, win: Window) -> pd.DataFrame:
    """hero_id, games and wins per hero for population rows in the window, by hero_id."""
    raise NotImplementedError


def add_hero_names(df: pd.DataFrame, heroes: pd.DataFrame) -> pd.DataFrame:
    """Add name and release_date from `heroes` (left join on hero_id).

    A hero missing from `heroes` is named "hero <id>". hero_id, name and release_date
    come first, the other columns keep their order.
    """
    raise NotImplementedError
