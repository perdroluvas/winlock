import argparse

import duckdb

from winlock.db import connect

PRIVATE_COLUMNS = {"account_id"}
DEFAULT_HEAD_COLUMNS = [
    "match_id",
    "hero_id",
    "team",
    "won",
    "start_time",
    "duration_s",
    "game_mode",
    "match_mode",
    "match_outcome",
]


def drop_private(rel: duckdb.DuckDBPyRelation) -> duckdb.DuckDBPyRelation:
    keep = [c for c in rel.columns if c not in PRIVATE_COLUMNS]
    if len(keep) < len(rel.columns):
        print(f"(hidden: {', '.join(sorted(PRIVATE_COLUMNS & set(rel.columns)))})")
    if not keep:
        raise SystemExit("Nothing to show: the query only returns private columns.")
    return rel.project(", ".join(f'"{c}"' for c in keep))


def cmd_schema(con: duckdb.DuckDBPyConnection, args: argparse.Namespace) -> None:
    query = "SELECT column_name, column_type FROM (DESCRIBE match_player)"
    if args.like:
        rel = con.sql(query + " WHERE column_name ILIKE ?", params=[f"%{args.like}%"])
    else:
        rel = con.sql(query)
    rel.show(max_rows=200)


def cmd_head(con: duckdb.DuckDBPyConnection, args: argparse.Namespace) -> None:
    cols = args.cols.split(",") if args.cols else DEFAULT_HEAD_COLUMNS
    select = ", ".join(f'"{c.strip()}"' for c in cols)
    rel = con.sql(f"SELECT {select} FROM match_player LIMIT {args.n}")
    drop_private(rel).show()


def cmd_sql(con: duckdb.DuckDBPyConnection, args: argparse.Namespace) -> None:
    drop_private(con.sql(args.query)).show(max_rows=args.max_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the match_player parquet with DuckDB.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_schema = sub.add_parser("schema", help="list columns and types")
    p_schema.add_argument("--like", help="only columns whose name contains this text")
    p_schema.set_defaults(func=cmd_schema)

    p_head = sub.add_parser("head", help="show the first rows")
    p_head.add_argument("-n", type=int, default=5, help="number of rows (default: 5)")
    p_head.add_argument("--cols", help="comma-separated columns (default: a key subset)")
    p_head.set_defaults(func=cmd_head)

    p_sql = sub.add_parser("sql", help="run a query against the `match_player` view")
    p_sql.add_argument("query")
    p_sql.add_argument("--max-rows", type=int, default=50, help="rows to print (default: 50)")
    p_sql.set_defaults(func=cmd_sql)

    args = parser.parse_args()
    args.func(connect(), args)


if __name__ == "__main__":
    main()
