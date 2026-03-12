"""Pytest configuration: ensure PCAB database is initialized before router tests."""

import sys
from pathlib import Path

import pytest

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dynamic_routing.pcab import (
    get_db_path,
    populate_mock_data,
    setup_pcab_database,
)


@pytest.fixture(scope="session", autouse=True)
def ensure_pcab_initialized():
    """Initialize PCAB database before any router tests run."""
    db_path = get_db_path()
    if not db_path.exists():
        conn = setup_pcab_database(db_path)
        populate_mock_data(conn)
        conn.close()
    else:
        # Ensure schema and data exist (idempotent)
        conn = setup_pcab_database(db_path)
        populate_mock_data(conn)
        conn.close()
