"""
Personalized Context Assembly Benchmark (PCAB): Database and agent tools.

SQLite-backed environment for reproducible evaluation. Use setup_pcab_database()
to initialize and populate the schema before running agents.
"""

import json
import re
import sqlite3
from pathlib import Path

# Default DB path: project_root/pcab_environment.db
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB_PATH = _PROJECT_ROOT / "pcab_environment.db"


def get_db_path() -> Path:
    """Return the path to the PCAB database."""
    return _DEFAULT_DB_PATH


def setup_pcab_database(db_path: str | Path | None = None) -> sqlite3.Connection:
    """
    Initialize the PCAB schema. Creates tables if they do not exist.
    Does not populate data; call populate_mock_data() separately.
    """
    path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    conn = sqlite3.connect(str(path))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS calendar (
            id INTEGER PRIMARY KEY,
            date TEXT,
            start_time TEXT,
            end_time TEXT,
            title TEXT,
            location TEXT,
            attendees TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY,
            due_date TEXT,
            priority TEXT,
            title TEXT,
            status TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS drive_docs (
            id INTEGER PRIMARY KEY,
            title TEXT,
            snippet TEXT,
            content TEXT,
            last_modified TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS commutes (
            origin TEXT,
            destination TEXT,
            transit_mode TEXT,
            estimated_minutes INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            name TEXT PRIMARY KEY,
            role TEXT,
            email TEXT,
            meeting_preferences TEXT
        )
    """)

    conn.commit()
    return conn


def populate_mock_data(conn: sqlite3.Connection | None = None) -> None:
    """
    Inject controlled benchmark data, including intentional noise.
    Call after setup_pcab_database(). Pass conn to reuse, or None to create new.
    """
    own_conn = conn is None
    if conn is None:
        conn = sqlite3.connect(str(_DEFAULT_DB_PATH))

    cursor = conn.cursor()

    tables = ["calendar", "tasks", "drive_docs", "commutes", "contacts"]
    for table in tables:
        cursor.execute(f"DELETE FROM {table}")

    calendar_data = [
        ("2026-03-16", "14:00", "14:50", "AAI 820", "Burchard Hall", "Class"),
        ("2026-03-16", "11:00", "11:30", "Quick Sync with Dr. Man", "Burchard Hall", "Dr. Hong Man"),
        ("2026-03-16", "13:00", "14:00", "AAI-800 Presentation Prep", "Library", "Self"),
    ]
    cursor.executemany(
        "INSERT INTO calendar (date, start_time, end_time, title, location, attendees) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        calendar_data,
    )

    task_data = [
        ("2026-03-16", "High", "Submit draft proposal to advisor", "Pending"),
        ("2026-03-17", "Medium", "Running tests on various agent architectures", "Pending"),
        ("2026-03-15", "Low", "Buy coffee beans", "Overdue"),
    ]
    cursor.executemany(
        "INSERT INTO tasks (due_date, priority, title, status) VALUES (?, ?, ?, ?)",
        task_data,
    )

    drive_data = [
        ("Project_Notes_V2", "Multi-agent scaling laws...", "Full text about DeepMind paper...", "2026-03-10"),
        ("Advisor_Meeting_Prep", "Ask about PCAB benchmark...", "Need to clarify Oracle baseline...", "2026-03-11"),
    ]
    cursor.executemany(
        "INSERT INTO drive_docs (title, snippet, content, last_modified) VALUES (?, ?, ?, ?)",
        drive_data,
    )

    commute_data = [
        ("Babbio Center", "Gateway North", "Walk", 8),
        ("Babbio Center", "Burchard Hall", "Walk", 8),
        ("Gateway North", "Library", "Walk", 5),
        ("Library", "Hoboken Coffee", "Walk", 12),
    ]
    cursor.executemany(
        "INSERT INTO commutes (origin, destination, transit_mode, estimated_minutes) "
        "VALUES (?, ?, ?, ?)",
        commute_data,
    )

    contact_data = [
        ("Dr. Hong Man", "Project Advisor", "hman@stevens.edu", "Prefers meetings before noon. Requires 24h notice."),
    ]
    cursor.executemany(
        "INSERT INTO contacts (name, role, email, meeting_preferences) VALUES (?, ?, ?, ?)",
        contact_data,
    )

    conn.commit()
    if own_conn:
        conn.close()


# --- Parameter extraction (shared by SAS and CMAS) ---
# When extraction_overrides is provided (from PCAB task registry), these values
# take precedence over keyword-based extraction—enabling data-driven benchmarks.


def extract_date(task: str, extraction_overrides: dict | None = None) -> str:
    """Extract date from task; default to benchmark date 2026-03-16."""
    if extraction_overrides and "date" in extraction_overrides:
        return str(extraction_overrides["date"])
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", task)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    if "march 16" in task.lower() or "mar 16" in task.lower() or "3/16" in task.lower():
        return "2026-03-16"
    return "2026-03-16"


def extract_drive_query(task: str, extraction_overrides: dict | None = None) -> str:
    """Extract search terms for Drive from task."""
    if extraction_overrides and "drive_query" in extraction_overrides:
        return str(extraction_overrides["drive_query"])
    task_lower = task.lower()
    if "capstone" in task_lower or "proposal" in task_lower:
        return "Capstone"
    if "advisor" in task_lower or "meeting prep" in task_lower:
        return "Advisor"
    if "notes" in task_lower:
        return "notes"
    return "Capstone"


def extract_commute_pair(task: str, extraction_overrides: dict | None = None) -> tuple[str, str] | None:
    """Extract origin/destination for commute. Returns (origin, dest) or None."""
    if extraction_overrides:
        origin = extraction_overrides.get("commute_origin")
        dest = extraction_overrides.get("commute_destination")
        if origin and dest:
            return (str(origin), str(dest))
    task_lower = task.lower()
    if "commute" not in task_lower and "walk" not in task_lower and "travel" not in task_lower:
        return None
    locs = ["babbio center", "gateway north", "library", "hoboken coffee", "burchard hall"]
    found = [loc for loc in locs if loc in task_lower]
    if len(found) >= 2:
        return (found[0].title(), found[1].title())
    if "advisor" in task_lower or "dr." in task_lower:
        return ("Babbio Center", "Gateway North")
    return ("Babbio Center", "Gateway North")


def extract_contact_name(task: str, extraction_overrides: dict | None = None) -> str | None:
    """Extract contact name from task."""
    if extraction_overrides and "contact_name" in extraction_overrides:
        return str(extraction_overrides["contact_name"])
    if "dr. hong man" in task.lower() or "dr hong man" in task.lower():
        return "Dr. Hong Man"
    if "advisor" in task.lower():
        return "Dr. Hong Man"
    return None


# --- Agent Tools (for LangGraph / LLM binding) ---


def get_calendar_events(date: str, db_path: str | Path | None = None) -> str:
    """Fetch all calendar events for a specific YYYY-MM-DD."""
    path = str(db_path) if db_path else str(_DEFAULT_DB_PATH)
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT start_time, end_time, title, location FROM calendar WHERE date = ?",
            (date,),
        )
        events = cursor.fetchall()

    if not events:
        return json.dumps({"status": "empty", "message": f"No events found for {date}."})

    result = [{"start": e[0], "end": e[1], "title": e[2], "location": e[3]} for e in events]
    return json.dumps(result)


def search_drive_docs(query: str, db_path: str | Path | None = None) -> str:
    """Search Drive notes for a keyword."""
    path = str(db_path) if db_path else str(_DEFAULT_DB_PATH)
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        search_term = f"%{query}%"
        cursor.execute(
            "SELECT title, snippet, last_modified FROM drive_docs WHERE title LIKE ? OR content LIKE ?",
            (search_term, search_term),
        )
        docs = cursor.fetchall()

    if not docs:
        return json.dumps({"status": "empty", "message": "No documents match that query."})

    result = [{"title": d[0], "snippet": d[1], "last_modified": d[2]} for d in docs]
    return json.dumps(result)


def estimate_commute(origin: str, destination: str, db_path: str | Path | None = None) -> str:
    """Get estimated travel time between two locations."""
    path = str(db_path) if db_path else str(_DEFAULT_DB_PATH)
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT transit_mode, estimated_minutes FROM commutes WHERE origin = ? AND destination = ?",
            (origin, destination),
        )
        route = cursor.fetchone()

    if not route:
        return json.dumps({"error": "Route not found in mapping database."})

    return json.dumps({
        "origin": origin,
        "destination": destination,
        "mode": route[0],
        "minutes": route[1],
    })


def get_contact_preferences(name: str, db_path: str | Path | None = None) -> str:
    """Look up a contact's email and meeting preferences."""
    path = str(db_path) if db_path else str(_DEFAULT_DB_PATH)
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, email, meeting_preferences FROM contacts WHERE name = ?",
            (name,),
        )
        contact = cursor.fetchone()

    if not contact:
        return json.dumps({"error": "Contact not found."})

    return json.dumps({
        "name": name,
        "role": contact[0],
        "email": contact[1],
        "preferences": contact[2],
    })
