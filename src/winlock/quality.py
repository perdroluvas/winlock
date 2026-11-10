"""Data-quality checks."""

import duckdb
import pandas as pd


def start_day_profile(con: duckdb.DuckDBPyConnection, recent_days: int = 3) -> pd.DataFrame:
    return con.sql(
        """
        WITH days AS (
            SELECT CAST(start_time AS DATE) AS day, match_id FROM match_player
        ),
        bucketed AS (
            SELECT
                CASE WHEN day > (SELECT MAX(day) FROM days) - $recent
                     THEN CAST(day AS VARCHAR)
                     ELSE 'older' END AS start_day,
                MIN(day) AS first_day,
                COUNT(*) AS rows,
                COUNT(DISTINCT match_id) AS matches
            FROM days
            GROUP BY 1
        )
        SELECT start_day, rows, matches,
               ROUND(100 * rows / SUM(rows) OVER (), 1) AS pct_rows
        FROM bucketed
        ORDER BY first_day
        """,
        params={"recent": recent_days},
    ).df()


def starts_per_bin(
    con: duckdb.DuckDBPyConnection, start: str, end: str, game_mode: str, minutes: int
) -> pd.DataFrame:
    """The update shows as a dip in match starts (game_mode_version never changes)."""
    return con.execute(
        """
        SELECT time_bucket(INTERVAL (CAST($minutes AS INTEGER)) MINUTE, start_time) AS bin_start,
               COUNT(DISTINCT match_id) AS matches
        FROM match_player
        WHERE start_time >= CAST($start AS TIMESTAMPTZ)
          AND start_time < CAST($end AS TIMESTAMPTZ)
          AND game_mode = $game_mode
        GROUP BY 1
        ORDER BY 1
        """,
        {"start": start, "end": end, "game_mode": game_mode, "minutes": minutes},
    ).df()


def match_completeness(
    con: duckdb.DuckDBPyConnection, start: str, end: str, game_mode: str
) -> pd.DataFrame:
    return con.sql(
        """
        WITH matches AS (
            SELECT match_id, COUNT(*) AS rows, SUM(CAST(won AS INT)) AS winners
            FROM match_player
            WHERE start_time >= CAST($start AS TIMESTAMPTZ)
              AND start_time < CAST($end AS TIMESTAMPTZ) AND game_mode = $game_mode
            GROUP BY match_id
        )
        SELECT rows, COUNT(*) AS matches, ROUND(AVG(winners), 2) AS avg_winners,
               ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_matches
        FROM matches
        GROUP BY rows
        ORDER BY rows DESC
        """,
        params={"start": start, "end": end, "game_mode": game_mode},
    ).df()
