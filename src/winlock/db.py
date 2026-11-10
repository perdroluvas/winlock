import duckdb
import pandas as pd

from winlock.config import load_config, resolve_path


def connect() -> duckdb.DuckDBPyConnection:
    cfg = load_config()
    path = resolve_path(cfg["data"]["match_player"])
    con = duckdb.connect()
    # Days and hours are defined in UTC, not in the machine's local time zone.
    con.execute("SET TimeZone = 'UTC'")
    con.execute(f"CREATE VIEW match_player AS SELECT * FROM read_parquet('{path}')")
    return con


def load_heroes() -> pd.DataFrame:
    cfg = load_config()
    return pd.read_csv(resolve_path(cfg["data"]["heroes"]), parse_dates=["release_date"])
