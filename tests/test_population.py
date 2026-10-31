import pandas as pd

from conftest import CFG
from winlock.population import add_hero_names, hero_winrates, population_report


def test_drops_incomplete_other_mode_and_other_day(ten_match_con):
    ids = {r[0] for r in ten_match_con.sql("SELECT DISTINCT match_id FROM population").fetchall()}
    assert ids == set(range(1, 11))


def test_temporal_split_is_70_30_by_time(ten_match_con):
    rows = ten_match_con.sql(
        "SELECT match_id, split FROM population GROUP BY ALL ORDER BY match_id"
    ).fetchall()
    assert [s for _, s in rows] == ["train"] * 7 + ["test"] * 3


def test_no_match_on_both_sides(ten_match_con):
    n = ten_match_con.execute(
        "SELECT COUNT(*) FROM (SELECT match_id FROM population "
        "GROUP BY match_id HAVING COUNT(DISTINCT split) > 1)"
    ).fetchall()[0][0]
    assert n == 0


def test_population_has_no_account_id(ten_match_con):
    assert "account_id" not in ten_match_con.sql("SELECT * FROM population").columns


def test_hero_winrates_counts_train_only(ten_match_con):
    wr = hero_winrates(ten_match_con, "train")
    assert wr["games"].tolist() == [7] * 12
    assert wr["wins"].sum() == 7 * 6


def test_report_ends_with_population_size(ten_match_con):
    report = population_report(ten_match_con, CFG)
    assert report["step"].iloc[-1] == "complete_match"
    assert report["matches"].iloc[-1] == 10


def test_add_hero_names_falls_back_for_unknown_id():
    heroes = pd.DataFrame(
        {"hero_id": [1], "name": ["Infernus"], "release_date": [pd.Timestamp("2024-04-24")]}
    )
    out = add_hero_names(pd.DataFrame({"hero_id": [1, 99], "games": [10, 5]}), heroes)
    assert out["name"].tolist() == ["Infernus", "hero 99"]
    assert list(out.columns[:3]) == ["hero_id", "name", "release_date"]


def test_parity_halves_cover_the_whole_population(ten_match_con):
    from winlock.population import hero_winrates_by_parity

    even, odd = hero_winrates_by_parity(ten_match_con, 0), hero_winrates_by_parity(ten_match_con, 1)
    assert even["games"].sum() + odd["games"].sum() == 10 * 12
    assert even["games"].sum() == 5 * 12  # match_ids 1..10: five even
