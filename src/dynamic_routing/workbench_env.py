"""
WorkBench sandbox path and tool loading.

WorkBench modules read CSV paths relative to the repository root; imports must
happen with cwd set to that root.
"""

from __future__ import annotations

import ast
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Project root: .../DynamicRoutingAgents
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def workbench_root() -> Path:
    env = os.environ.get("WORKBENCH_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (_PROJECT_ROOT / "vendor" / "WorkBench").resolve()


@contextmanager
def workbench_session():
    """Temporarily chdir to WorkBench root and prepend it to sys.path."""
    root = workbench_root()
    if not root.is_dir():
        raise FileNotFoundError(
            f"WorkBench not found at {root}. Clone: python scripts/setup_workbench.py"
        )
    prev_cwd = os.getcwd()
    prev_path = sys.path.copy()
    os.chdir(root)
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    try:
        yield root
    finally:
        os.chdir(prev_cwd)
        sys.path[:] = prev_path


def reset_workbench_state() -> None:
    """Reload all sandbox DataFrames from disk (fresh state per task/architecture)."""
    with workbench_session():
        from src.tools import (
            analytics,
            calendar,
            customer_relationship_manager,
            email,
            project_management,
        )

        for mod in (
            calendar,
            email,
            analytics,
            project_management,
            customer_relationship_manager,
        ):
            mod.reset_state()


def parse_domains_cell(domains_cell: str) -> list[str]:
    """Parse the CSV `domains` column, e.g. \"['email']\" -> [\"email\"]."""
    val: Any = ast.literal_eval(domains_cell)
    if isinstance(val, str):
        return [val]
    return list(val)


def get_tools_for_domains(domains: list[str]) -> list:
    """
    Return LangChain tools for the given WorkBench domains, plus company directory
    (matches upstream get_toolkits behavior).
    """
    with workbench_session():
        from src.tools.toolkits import (
            analytics_toolkit,
            calendar_toolkit,
            company_directory_toolkit,
            customer_relationship_manager_toolkit,
            email_toolkit,
            project_management_toolkit,
        )

        tools: list = []
        if "email" in domains:
            tools.extend(email_toolkit)
        if "calendar" in domains:
            tools.extend(calendar_toolkit)
        if "analytics" in domains:
            tools.extend(analytics_toolkit)
        if "project_management" in domains:
            tools.extend(project_management_toolkit)
        if "customer_relationship_manager" in domains:
            tools.extend(customer_relationship_manager_toolkit)
        tools.extend(company_directory_toolkit)
        return tools


def workbench_system_prompt_prefix() -> str:
    """Same date context string WorkBench uses in generate_results."""
    with workbench_session():
        from src.data_generation.data_generation_utils import HARDCODED_CURRENT_TIME

        dt = HARDCODED_CURRENT_TIME
        return (
            f"Today is {dt.strftime('%A')}, {dt.date()} and the current time is {dt.time()}. "
            "Remember the current date and time when answering queries. "
            "Meetings must not start before 9am or end after 6pm.\n\n"
        )
