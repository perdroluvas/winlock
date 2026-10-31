"""Shared fixtures: a tiny fake match_player table in an in-memory DuckDB."""

from datetime import UTC, datetime, timedelta

import duckdb
import pandas as pd
import pytest

from winlock.population import create_population

CFG = {
    "population": {
        "game_mode": "Normal",
        "match_mode": ["Unranked", "Ranked"],
        "match_outcome": "TeamWin",
        "start_day": "2026-10-20",
        "players_per_match": 12,
        "wins_per_match": 6,
    },
    "split": {"method": "temporal", "train_fraction": 0.7},
}
DAY = datetime(2026, 10, 20, tzinfo=UTC)


def match_rows(match_id, start, players=12, game_mode="Normal"):
    """Rows of one fake match: the first half of the players win."""
    return [
        {
            "match_id": match_id,
            "account_id": 1000 * match_id + i,
            "hero_id": i + 1,
            "team": "Team0" if i < players // 2 else "Team1",
            "won": i < players // 2,
            "start_time": start,
            "game_mode": game_mode,
            "match_mode": "Unranked",
            "match_outcome": "TeamWin",
        }
        for i in range(players)
    ]


def make_con(rows):
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    df = pd.DataFrame(rows)  # noqa: F841 (read by DuckDB by name)
    con.execute("CREATE TABLE match_player AS SELECT * FROM df")
    return con


@pytest.fixture
def ten_match_con():
    rows = []
    for k in range(10):
        rows += match_rows(match_id=k + 1, start=DAY + timedelta(hours=k))
    rows += match_rows(match_id=100, start=DAY, players=10)  # incomplete
    rows += match_rows(match_id=200, start=DAY, players=8, game_mode="StreetBrawl")
    rows += match_rows(match_id=300, start=DAY - timedelta(days=3))  # other day
    con = make_con(rows)
    create_population(con, CFG)
    return con
