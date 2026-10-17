"""Build the DuckDB database by running every sql/*.sql file in order.

Usage:
    python -m coachmodel.build_db
"""

from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = REPO_ROOT / "sql"
DATABASE_PATH = REPO_ROOT / "data" / "coach.duckdb"


def build(database_path: Path = DATABASE_PATH, data_root: Path = REPO_ROOT) -> duckdb.DuckDBPyConnection:
    """Run the SQL pipeline. SQL files read data/raw/... relative to data_root, so tests
    can point it at a small fixture folder instead of the full download."""
    connection = duckdb.connect(str(database_path))
    connection.execute(f"SET file_search_path = '{data_root}'")
    for sql_file in sorted(SQL_DIR.glob("*.sql")):
        print(f"running {sql_file.name}")
        connection.execute(sql_file.read_text())
    return connection


def main() -> None:
    connection = build()
    for table in ["games", "pbp", "decision_events", "team_state", "decision_state", "stoppages", "stoppage_players"]:
        row_count = connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"{table}: {row_count:,} rows")
    connection.close()


if __name__ == "__main__":
    main()
