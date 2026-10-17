from pathlib import Path

import pytest

from coachmodel.build_db import build

FIXTURE_ROOT = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def database(tmp_path_factory):
    """The full SQL pipeline built from the three saved fixture games."""
    connection = build(tmp_path_factory.mktemp("db") / "test.duckdb", data_root=FIXTURE_ROOT)
    yield connection
    connection.close()
