from datetime import UTC, datetime

import pandas as pd

from conftest import CFG, TRAIN, make_con, match_rows
from winlock.population import (
    add_hero_names,
    create_population,
    hero_winrates,
    population_report,
    population_span,
    window,
    window_sizes,
)


def ids(con, where="TRUE"):
    return {
        r[0] for r in con.sql(f"SELECT DISTINCT match_id FROM population WHERE {where}").fetchall()
    }


def test_drops_incomplete_other_mode_and_out_of_window(ten_match_con):
    assert ids(ten_match_con) == set(range(1, 11))


def test_windows_and_span():
    assert window(CFG, "train") == ("2026-10-20T00:00:00Z", "2026-10-20T07:00:00Z")
    assert population_span(CFG) == ("2026-10-20T00:00:00Z", "2026-10-20T10:00:00Z")


def test_window_sizes_split_by_time(ten_match_con):
    sizes = window_sizes(ten_match_con, CFG)
    assert sizes["window"].tolist() == ["train", "test"]
    assert sizes["matches"].tolist() == [7, 3]


def test_population_has_no_account_id(ten_match_con):
    assert "account_id" not in ten_match_con.sql("SELECT * FROM population").columns


def test_hero_winrates_counts_one_window(ten_match_con):
    wr = hero_winrates(ten_match_con, TRAIN)
    assert wr["games"].tolist() == [7] * 12
    assert wr["wins"].sum() == 7 * 6


def test_report_ends_with_population_size(ten_match_con):
    report = population_report(ten_match_con, CFG)
    assert report["step"].tolist() == [
        "file",
        "window",
        "game_mode",
        "match_mode",
        "match_outcome",
        "complete_match",
    ]
    assert report["matches"].iloc[-1] == 10


def test_add_hero_names_falls_back_for_unknown_id():
    heroes = pd.DataFrame(
        {"hero_id": [1], "name": ["Infernus"], "release_date": [pd.Timestamp("2024-04-24")]}
    )
    out = add_hero_names(pd.DataFrame({"hero_id": [1, 99], "games": [10, 5]}), heroes)
    assert out["name"].tolist() == ["Infernus", "hero 99"]
    assert list(out.columns[:3]) == ["hero_id", "name", "release_date"]


def window_con(exclude_flags=()):
    """Fake matches around a fake update at 10-30 21:00: two before, one in the buffer, two
    after (one incomplete), one in StreetBrawl, and one with a low-priority player."""

    def at(hour):
        return datetime(2026, 10, 30, hour, tzinfo=UTC)

    rows = (
        match_rows(1, at(10))
        + match_rows(2, at(19))
        + match_rows(3, at(21))  # buffer 20:30-21:30
        + match_rows(4, at(22))
        + match_rows(5, at(23), players=11)  # incomplete
        + match_rows(6, at(22), players=8, game_mode="StreetBrawl")
        + match_rows(7, at(23))
    )
    for r in rows:
        r["low_pri_pool"] = r["match_id"] == 7 and r["hero_id"] == 1  # one flagged player
    con = make_con(rows)
    cfg = {
        "population": {**CFG["population"], "exclude_flags": list(exclude_flags)},
        "windows": {
            "pre": ["2026-10-30T00:00:00Z", "2026-10-30T20:30:00Z"],
            "post": ["2026-10-30T21:30:00Z", "2026-10-31T00:00:00Z"],
        },
    }
    create_population(con, cfg)
    return con, cfg


def test_window_population_keeps_the_buffer_out_of_every_window():
    con, cfg = window_con()
    assert ids(con) == {1, 2, 3, 4, 7}  # 3 is inside the span, between the windows
    assert window_sizes(con, cfg)["matches"].tolist() == [2, 2]
    after = hero_winrates(con, window(cfg, "post"))
    assert after["games"].tolist() == [2] * 12


def test_exclude_flags_drop_the_whole_match():
    con, _ = window_con(exclude_flags=["low_pri_pool"])
    assert ids(con) == {1, 2, 3, 4}  # match 7 lost one player, so it is no longer complete
