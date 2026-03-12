#!/usr/bin/env python3
"""
Initialize the PCAB database with schema and gold trajectory data.

Run from project root:
    python scripts/setup_pcab.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from dynamic_routing.pcab import setup_pcab_database, populate_mock_data, get_db_path


def main() -> None:
    db_path = get_db_path()
    conn = setup_pcab_database(db_path)
    populate_mock_data(conn)
    conn.close()
    print(f"PCAB database initialized and populated at {db_path}")


if __name__ == "__main__":
    main()
