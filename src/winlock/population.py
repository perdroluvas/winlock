"""Filters that define the analysis population and the train/test split."""

from typing import Any

import duckdb
import pandas as pd

# Only the columns the analysis needs; account_id never enters the population table.
POPULATION_COLUMNS = ["match_id", "hero_id", "team", "won", "start_time"]

# Row filters, applied cumulatively (in this order) by population_report.
ROW_FILTERS = [
    ("start_day", "CAST(start_time AS DATE) = CAST($start_day AS DATE)"),
    ("game_mode", "game_mode = $game_mode"),
    ("match_mode", "list_contains($match_mode, match_mode)"),
    ("match_outcome", "match_outcome = $match_outcome"),
]


def _filter_params(cfg: dict[str, Any]) -> dict[str, Any]:
    pop = cfg["population"]
    return {key: pop[key] for key, _ in ROW_FILTERS}


def create_population(con: duckdb.DuckDBPyConnection, cfg: dict[str, Any]) -> None:
    """Create the `population` table: filtered rows of complete matches, each tagged with its split.

    Steps: (1) row filters from the config, (2) keep only matches with exactly
    `players_per_match` rows and `wins_per_match` winners, (3) rank matches by start_time and put
    the first `train_fraction` of them in 'train', the rest in 'test'. The split is decided per
    match, so all players of a match land on the same side.
    """
    pop, split = cfg["population"], cfg["split"]
    if split["method"] != "temporal":
        raise ValueError(f"Unknown split method: {split['method']!r}")
    where = " AND ".join(cond for _, cond in ROW_FILTERS)
    cols = ", ".join(f"f.{c}" for c in POPULATION_COLUMNS)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE population AS
        WITH filtered AS (
            SELECT * FROM match_player WHERE {where}
        ),
        complete AS (
            SELECT match_id, MIN(start_time) AS match_start
            FROM filtered
            GROUP BY match_id
            HAVING COUNT(*) = $players AND SUM(CAST(won AS INT)) = $wins
        ),
        ranked AS (
            SELECT match_id,
                   ROW_NUMBER() OVER (ORDER BY match_start, match_id) AS rank,
                   COUNT(*) OVER () AS n_matches
            FROM complete
        )
        SELECT {cols},
               CASE WHEN r.rank <= $train_fraction * r.n_matches
                    THEN 'train' ELSE 'test' END AS split
        FROM filtered f
        JOIN ranked r USING (match_id)
        """,
        {
            **_filter_params(cfg),
            "players": pop["players_per_match"],
            "wins": pop["wins_per_match"],
            "train_fraction": split["train_fraction"],
        },
    )


def population_report(con: duckdb.DuckDBPyConnection, cfg: dict[str, Any]) -> pd.DataFrame:
    """Rows and matches left after each filter step (requires `create_population` first)."""
    steps = [("file", "SELECT COUNT(*), COUNT(DISTINCT match_id) FROM match_player", {})]
    params = _filter_params(cfg)
    conds: list[str] = []
    for key, cond in ROW_FILTERS:
        conds.append(cond)
        used = {k: params[k] for k, _ in ROW_FILTERS[: len(conds)]}
        query = (
            "SELECT COUNT(*), COUNT(DISTINCT match_id) FROM match_player "
            f"WHERE {' AND '.join(conds)}"
        )
        steps.append((key, query, used))
    steps.append(
        ("complete_match", "SELECT COUNT(*), COUNT(DISTINCT match_id) FROM population", {})
    )

    rows = []
    for step, query, step_params in steps:
        [(n_rows, n_matches)] = con.execute(query, step_params).fetchall()
        rows.append({"step": step, "rows": n_rows, "matches": n_matches})
    report = pd.DataFrame(rows)
    report["pct_matches_kept"] = (100 * report["matches"] / report["matches"].shift()).round(1)
    return report


def split_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Matches and rows per split. Sizes only (no outcomes), so it is safe for the test side."""
    return con.sql(
        """
        SELECT split, COUNT(DISTINCT match_id) AS matches, COUNT(*) AS rows,
               MIN(start_time) AS first_start, MAX(start_time) AS last_start
        FROM population
        GROUP BY split
        ORDER BY first_start
        """
    ).df()


def hero_winrates(con: duckdb.DuckDBPyConnection, split: str) -> pd.DataFrame:
    """Games and wins per hero on one side of the split."""
    return con.execute(
        """
        SELECT hero_id, COUNT(*) AS games, SUM(CAST(won AS INT)) AS wins
        FROM population
        WHERE split = $split
        GROUP BY hero_id
        ORDER BY hero_id
        """,
        {"split": split},
    ).df()


def hero_winrates_by_parity(con: duckdb.DuckDBPyConnection, parity: int) -> pd.DataFrame:
    """Games and wins per hero over the whole population, for matches with match_id % 2 = parity.

    Split-half control: even and odd match_ids cover the same hours, so any difference between
    the halves is chance alone (no time effect).
    """
    return con.execute(
        """
        SELECT hero_id, COUNT(*) AS games, SUM(CAST(won AS INT)) AS wins
        FROM population
        WHERE match_id % 2 = $parity
        GROUP BY hero_id
        ORDER BY hero_id
        """,
        {"parity": parity},
    ).df()


def add_hero_names(df: pd.DataFrame, heroes: pd.DataFrame) -> pd.DataFrame:
    """Put `name` and `release_date` right after `hero_id`; unknown ids become 'hero <id>'."""
    out = df.merge(heroes[["hero_id", "name", "release_date"]], on="hero_id", how="left")
    out["name"] = out["name"].fillna("hero " + out["hero_id"].astype(str))
    first = ["hero_id", "name", "release_date"]
    return out[first + [c for c in out.columns if c not in first]]
