"""Data-quality checks on the match_player view."""

import duckdb
import pandas as pd


def start_day_profile(con: duckdb.DuckDBPyConnection, recent_days: int = 3) -> pd.DataFrame:
    """Rows and matches per UTC start day: the last `recent_days` before the newest day are
    listed individually, everything older is grouped into one bucket.

    The file is cut by created_at (when a row reached the database), so late-arriving old
    matches show up as a tail of start_time values far before the file's window.
    """
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


def start_hour_profile(con: duckdb.DuckDBPyConnection, day: str) -> pd.DataFrame:
    """Rows and matches per UTC start hour on one day (format 'YYYY-MM-DD')."""
    return con.sql(
        """
        SELECT HOUR(start_time) AS hour_utc,
               COUNT(*) AS rows,
               COUNT(DISTINCT match_id) AS matches
        FROM match_player
        WHERE CAST(start_time AS DATE) = CAST($day AS DATE)
        GROUP BY 1
        ORDER BY 1
        """,
        params={"day": day},
    ).df()


def match_completeness(con: duckdb.DuckDBPyConnection, day: str, game_mode: str) -> pd.DataFrame:
    """How many matches have each number of rows, and their average number of winners.

    A complete match has 12 rows and 6 winners; anything else is missing players.
    """
    return con.sql(
        """
        WITH matches AS (
            SELECT match_id, COUNT(*) AS rows, SUM(CAST(won AS INT)) AS winners
            FROM match_player
            WHERE CAST(start_time AS DATE) = CAST($day AS DATE) AND game_mode = $game_mode
            GROUP BY match_id
        )
        SELECT rows, COUNT(*) AS matches, ROUND(AVG(winners), 2) AS avg_winners,
               ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_matches
        FROM matches
        GROUP BY rows
        ORDER BY rows DESC
        """,
        params={"day": day, "game_mode": game_mode},
    ).df()
