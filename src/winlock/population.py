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
    start, end = cfg["windows"][name]
    return start, end


def population_span(cfg: dict[str, Any]) -> Window:
    starts, ends = zip(*cfg["windows"].values(), strict=True)
    return min(starts), max(ends)


def _filter_params(cfg: dict[str, Any]) -> dict[str, Any]:
    pop = cfg["population"]
    start, end = population_span(cfg)
    return {"start": start, "end": end, **{key: pop[key] for key, _ in ROW_FILTERS[1:]}}


def create_population(con: duckdb.DuckDBPyConnection, cfg: dict[str, Any]) -> None:
    """Complete matches only (12 rows, 6 winners), dropped whole otherwise."""
    pop = cfg["population"]
    where = " AND ".join(cond for _, cond in ROW_FILTERS)
    flags = "".join(f" AND NOT {flag}" for flag in pop.get("exclude_flags", []))
    cols = ", ".join(f"f.{c}" for c in POPULATION_COLUMNS)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE population AS
        WITH filtered AS (
            SELECT * FROM match_player WHERE {where}{flags}
        ),
        complete AS (
            SELECT match_id FROM filtered
            GROUP BY match_id
            HAVING COUNT(*) = $players AND SUM(CAST(won AS INT)) = $wins
        )
        SELECT {cols} FROM filtered f JOIN complete USING (match_id)
        """,
        {**_filter_params(cfg), "players": pop["players_per_match"], "wins": pop["wins_per_match"]},
    )


def population_report(con: duckdb.DuckDBPyConnection, cfg: dict[str, Any]) -> pd.DataFrame:
    steps = [("file", "SELECT COUNT(*), COUNT(DISTINCT match_id) FROM match_player", {})]
    params = _filter_params(cfg)
    names = {"window": ["start", "end"]}
    conds: list[str] = []
    used: dict[str, Any] = {}
    for key, cond in ROW_FILTERS:
        conds.append(cond)
        used |= {k: params[k] for k in names.get(key, [key])}
        query = (
            "SELECT COUNT(*), COUNT(DISTINCT match_id) FROM match_player "
            f"WHERE {' AND '.join(conds)}"
        )
        steps.append((key, query, dict(used)))
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


def in_window(win: Window) -> str:
    start, end = win
    return f"start_time >= TIMESTAMPTZ '{start}' AND start_time < TIMESTAMPTZ '{end}'"


def window_sizes(con: duckdb.DuckDBPyConnection, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name in cfg["windows"]:
        [(n, first, last)] = con.execute(
            "SELECT COUNT(DISTINCT match_id), strftime(MIN(start_time), '%m-%d %H:%M'), "
            "strftime(MAX(start_time), '%m-%d %H:%M') FROM population "
            f"WHERE {in_window(window(cfg, name))}"
        ).fetchall()
        rows.append({"window": name, "matches": n, "first_start": first, "last_start": last})
    return pd.DataFrame(rows)


def hero_winrates(con: duckdb.DuckDBPyConnection, win: Window) -> pd.DataFrame:
    return con.execute(
        f"""
        SELECT hero_id, COUNT(*) AS games, SUM(CAST(won AS INT)) AS wins
        FROM population
        WHERE {in_window(win)}
        GROUP BY hero_id
        ORDER BY hero_id
        """
    ).df()


def add_hero_names(df: pd.DataFrame, heroes: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(heroes[["hero_id", "name", "release_date"]], on="hero_id", how="left")
    out["name"] = out["name"].fillna("hero " + out["hero_id"].astype(str))
    first = ["hero_id", "name", "release_date"]
    return out[first + [c for c in out.columns if c not in first]]
