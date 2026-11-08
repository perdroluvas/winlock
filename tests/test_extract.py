from datetime import UTC, datetime

import duckdb
import pandas as pd
import pytest

from winlock.extract import (
    base_partitions,
    current_files,
    deduplicate,
    ingestion_gap_hours,
    parse_utc,
    partitions_for_window,
    read_window,
    select_files,
    summarize,
)

START = datetime(2026, 10, 28, tzinfo=UTC)
END = datetime(2026, 11, 1, tzinfo=UTC)
T = START.timestamp()

MANIFEST = {
    "tables": {
        "match_player": {
            "generation": 3,
            "watermark_hi": END.timestamp() - 3 * 3600,
            "files": [
                {"key": "p5-1", "kind": "base", "generation": 3, "partition": 5, "hi": T},
                {"key": "p5-0", "kind": "base", "generation": 3, "partition": 5, "hi": T},
                {"key": "p6-0", "kind": "base", "generation": 3, "partition": 6, "hi": T},
                {"key": "p7-0", "kind": "base", "generation": 3, "partition": 7, "hi": T},
                {"key": "old-gen", "kind": "base", "generation": 2, "partition": 6, "hi": T},
                {
                    "key": "late",
                    "kind": "residual",
                    "generation": 3,
                    "lo": T,
                    "hi": T + 9e4,
                    "rows_by_partition": {"1": 12, "6": 24},
                },
                {
                    "key": "other",
                    "kind": "delta",
                    "generation": 3,
                    "lo": T,
                    "hi": T + 9e4,
                    "rows_by_partition": {"1": 12, "7": 12},
                },
                {
                    "key": "too-old",
                    "kind": "delta",
                    "generation": 3,
                    "lo": 0,
                    "hi": T - 1,
                    "rows_by_partition": {"6": 12},
                },
            ],
        }
    }
}


def test_parse_utc():
    assert parse_utc("2026-10-28T00:00:00Z") == START


def test_current_files_and_partitions():
    files = current_files(MANIFEST, "match_player")
    assert "old-gen" not in [f["key"] for f in files]
    parts = base_partitions(files)
    assert list(parts) == [5, 6, 7]
    assert [f["key"] for f in parts[5]] == ["p5-0", "p5-1"]  # first file first


def test_partitions_for_window_binary_search():
    # partition -> first start_time; the window [10-28, 11-01) needs 126..128
    starts = {
        p: parse_utc(t)
        for p, t in {
            124: "2026-10-24T21:56:00Z",
            125: "2026-10-26T03:34:00Z",
            126: "2026-10-27T18:55:00Z",
            127: "2026-10-29T06:39:00Z",
            128: "2026-10-30T21:26:00Z",
            129: "2026-11-01T07:56:00Z",
            130: "2026-11-02T19:22:00Z",
            131: "2026-11-04T07:31:00Z",
        }.items()
    }
    called = []

    def start_of(p):
        called.append(p)
        return starts[p]

    assert partitions_for_window(list(starts), start_of, START, END) == [126, 127, 128]
    assert len(called) < len(starts)  # binary search: not every footer is read


def test_select_files_by_partition_and_time():
    files = current_files(MANIFEST, "match_player")
    keys = [f["key"] for f in select_files(files, keep=[6], start=START)]
    assert keys == ["p6-0", "late"]  # 'other' holds no rows of 6; 'too-old' closed before START


def test_ingestion_gap_hours():
    assert ingestion_gap_hours(MANIFEST, "match_player", END) == pytest.approx(3)


def ts(text):
    return pd.Timestamp(text, tz="UTC")


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute("SET TimeZone = 'UTC'")
    return c


@pytest.fixture
def sources(tmp_path, con):
    """Two files: a base and a delta that repeats one player row with a newer created_at."""
    base = pd.DataFrame(
        {
            "match_id": [1, 1, 2, 9],
            "player_slot": [0, 1, 0, 0],
            "account_id": [0, 0, 7, 8],  # two anonymous players in match 1
            "hero_id": [10, 11, 12, 13],
            "won": [True, False, True, True],
            "start_time": [
                ts("2026-10-29 10:00"),
                ts("2026-10-29 10:00"),
                ts("2026-10-30 12:00"),
                ts("2026-10-20 12:00"),
            ],  # match 9: too old
            "created_at": [ts("2026-10-29 11:00")] * 2 + [ts("2026-10-30 13:00")] * 2,
            "extra": [1, 2, 3, 4],  # a column we do not ask for
        }
    )
    delta = base.iloc[[2]].assign(won=False, created_at=ts("2026-10-30 15:00"))
    paths = [tmp_path / "base.parquet", tmp_path / "delta.parquet"]
    for df, path in zip([base, delta], paths, strict=True):
        con.register("df", df)
        con.execute(f"COPY df TO '{path}' (FORMAT parquet)")
        con.unregister("df")
    return [str(p) for p in paths]


COLUMNS = ["match_id", "player_slot", "account_id", "hero_id", "won", "start_time", "created_at"]


def test_read_window_filters_time_and_columns(con, sources, tmp_path):
    n = read_window(con, sources, COLUMNS, START, END, tmp_path / "raw.parquet")
    assert n == 4  # match 1 (2 rows) + match 2 twice (base + delta); match 9 is out of window
    cols = con.execute(f"SELECT * FROM read_parquet('{tmp_path / 'raw.parquet'}')").description
    assert [c[0] for c in cols] == COLUMNS


def test_deduplicate_keeps_newest_and_both_anonymous_players(con, sources, tmp_path):
    raw, final = tmp_path / "raw.parquet", tmp_path / "final.parquet"
    read_window(con, sources, COLUMNS, START, END, raw)
    assert deduplicate(con, raw, final) == 3
    rows = con.execute(
        f"SELECT match_id, player_slot, won FROM read_parquet('{final}') ORDER BY ALL"
    ).fetchall()
    # match 1 keeps both anonymous players; match 2 keeps the newer (delta) row, won = False
    assert rows == [(1, 0, True), (1, 1, False), (2, 0, False)]
    s = summarize(con, final)
    assert s["rows"] == 3 and s["matches"] == 2


def test_deduplicate_keeps_first_arrival_time(con, sources, tmp_path):
    raw, final = tmp_path / "raw.parquet", tmp_path / "final.parquet"
    read_window(con, sources, COLUMNS, START, END, raw)
    deduplicate(con, raw, final)
    fmt = "'%m-%d %H:%M'"
    row = con.execute(
        f"""SELECT strftime(created_at, {fmt}), strftime(first_created_at, {fmt})
            FROM read_parquet('{final}') WHERE match_id = 2"""
    ).fetchall()[0]
    # match 2 arrived at 13:00 (base) and was re-sent at 15:00 (delta)
    assert row == ("10-30 15:00", "10-30 13:00")
