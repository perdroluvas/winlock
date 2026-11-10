"""`winlock extract`: read a time window of match_player from the public data dumps."""

import json
import math
import urllib.request
from bisect import bisect_left, bisect_right
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

USER_AGENT = "winlock (personal data project)"


def fetch_manifest(url: str, timeout: float = 30) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def parse_utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


def current_files(manifest: dict[str, Any], table: str) -> list[dict[str, Any]]:
    t = manifest["tables"][table]
    gen = t.get("generation")
    return [f for f in t["files"] if f.get("generation", gen) == gen]


def base_partitions(files: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    parts: dict[int, list[dict[str, Any]]] = {}
    for f in files:
        if f["kind"] == "base":
            parts.setdefault(f["partition"], []).append(f)
    return {p: sorted(parts[p], key=lambda f: f["key"]) for p in sorted(parts)}


def partitions_for_window(
    partitions: list[int], start_of: Callable[[int], datetime], start: datetime, end: datetime
) -> list[int]:
    """Binary search: match_ids grow with time, so partitions are sorted by start."""
    lo = max(bisect_right(partitions, start, key=start_of) - 1, 0)
    hi = bisect_left(partitions, end, key=start_of)
    return partitions[lo:hi]


def select_files(
    files: list[dict[str, Any]], keep: list[int], start: datetime
) -> list[dict[str, Any]]:
    wanted = {str(p) for p in keep}
    cutoff = start.timestamp()
    out = []
    for f in files:
        if f["kind"] == "base":
            if f["partition"] in keep:
                out.append(f)
        elif f.get("hi", math.inf) >= cutoff and (
            "rows_by_partition" not in f or wanted & set(f["rows_by_partition"])
        ):
            out.append(f)
    return out


def footer_min_start(con: duckdb.DuckDBPyConnection, url: str) -> datetime:
    text = con.execute(
        f"""
        SELECT MIN(stats_min_value) FROM parquet_metadata('{url}')
        WHERE path_in_schema = 'start_time'
        """
    ).fetchall()[0][0]
    return parse_utc(text.replace(" ", "T"))


def ingestion_gap_hours(manifest: dict[str, Any], table: str, end: datetime) -> float:
    return (end.timestamp() - manifest["tables"][table]["watermark_hi"]) / 3600


def _sql_list(items: list[str]) -> str:
    return "[" + ", ".join("'" + s.replace("'", "''") + "'" for s in items) + "]"


def read_window(
    con: duckdb.DuckDBPyConnection,
    sources: list[str],
    columns: list[str],
    start: datetime,
    end: datetime,
    out_path: Path,
) -> int:
    cols = ", ".join(f'"{c}"' for c in columns)
    con.execute(
        f"""
        COPY (
            SELECT {cols}
            FROM read_parquet({_sql_list(sources)}, union_by_name = true)
            WHERE start_time >= TIMESTAMPTZ '{start.isoformat()}'
              AND start_time < TIMESTAMPTZ '{end.isoformat()}'
        ) TO '{out_path}' (FORMAT parquet, COMPRESSION zstd)
        """
    )
    return con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchall()[0][0]


def deduplicate(con: duckdb.DuckDBPyConnection, raw_path: Path, out_path: Path) -> int:
    """One row per (match_id, player_slot): anonymous players all have account_id 0."""
    con.execute(
        f"""
        COPY (
            SELECT *,
                   MIN(created_at) OVER (PARTITION BY match_id, player_slot) AS first_created_at
            FROM read_parquet('{raw_path}')
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY match_id, player_slot ORDER BY created_at DESC
            ) = 1
            ORDER BY start_time, match_id, player_slot
        ) TO '{out_path}' (FORMAT parquet, COMPRESSION zstd)
        """
    )
    return con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchall()[0][0]


def summarize(con: duckdb.DuckDBPyConnection, path: Path) -> dict[str, Any]:
    fmt = "'%Y-%m-%dT%H:%M:%SZ'"
    r = con.execute(
        f"""
        SELECT COUNT(*), COUNT(DISTINCT match_id),
               strftime(MIN(start_time), {fmt}), strftime(MAX(start_time), {fmt}),
               strftime(MIN(created_at), {fmt}), strftime(MAX(created_at), {fmt})
        FROM read_parquet('{path}')
        """
    ).fetchall()[0]
    keys = ["rows", "matches", "start_min", "start_max", "created_min", "created_max"]
    return dict(zip(keys, r, strict=True))


def run_extract(cfg: dict[str, Any], out_root: Path, dry_run: bool = False) -> dict[str, Any]:
    ex = cfg["extract"]
    start, end = parse_utc(ex["window_start"]), parse_utc(ex["window_end"])
    manifest = fetch_manifest(ex["manifest_url"])
    t = manifest["tables"][ex["table"]]
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    con.execute(f"SET http_retries = {ex['http_retries']}")

    def url(f: dict[str, Any]) -> str:
        return f"{manifest['public_url']}/{f['key']}"

    files = current_files(manifest, ex["table"])
    parts = base_partitions(files)
    probed: dict[int, str] = {}

    def start_of(p: int) -> datetime:
        first = footer_min_start(con, url(parts[p][0]))
        probed[p] = first.isoformat()
        return first

    keep = partitions_for_window(list(parts), start_of, start, end)
    files = select_files(files, keep, start)
    meta: dict[str, Any] = {
        "manifest_version": manifest["version"],
        "manifest_generated_at": manifest["generated_at"],
        "generation": t.get("generation"),
        "watermark_hi": datetime.fromtimestamp(t["watermark_hi"], UTC).isoformat(),
        "ingestion_gap_hours": round(ingestion_gap_hours(manifest, ex["table"], end), 2),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "columns": ex["columns"],
        "partitions": keep,
        "partition_starts_probed": dict(sorted(probed.items())),
        "files": [f["key"] for f in files],
        "files_by_kind": {
            k: sum(f["kind"] == k for f in files) for k in {f["kind"] for f in files}
        },
        "files_gb": round(sum(f.get("bytes", 0) for f in files) / 1e9, 2),
    }
    if dry_run:
        return meta

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%MZ")
    out_dir = out_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    raw, final = out_dir / "raw.parquet", out_dir / "match_player.parquet"
    urls = [url(f) for f in files]
    meta["extracted_at"] = datetime.now(UTC).isoformat()
    meta["rows_raw"] = read_window(con, urls, ex["columns"], start, end, raw)
    meta["rows_final"] = deduplicate(con, raw, final)
    meta["duplicates_removed"] = meta["rows_raw"] - meta["rows_final"]
    meta["summary"] = summarize(con, final)
    raw.unlink()
    meta["output"] = str(final)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta
