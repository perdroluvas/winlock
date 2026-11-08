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


def starts_per_bin(
    con: duckdb.DuckDBPyConnection, start: str, end: str, game_mode: str, minutes: int
) -> pd.DataFrame:
    """Matches of one mode started per `minutes`-long bin in [start, end) (UTC, ISO strings).

    Around an update, servers restart and players wait, so starts dip: that dip is how the
    update shows up in the data (`game_mode_version` does not change).
    """
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
    """Matches started in [start, end): how many have each number of rows, and their average
    number of winners.

    A complete match has 12 rows and 6 winners; anything else is missing players.
    """
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
