"""Load a ``.env`` file from the project root.

Preference order:
1. ``python-dotenv`` if available (canonical).
2. Minimal built-in parser (so benchmarks still read your API keys from ``.env``
   even if ``python-dotenv`` is missing from the active Python env).

Silent failure was a past footgun — now we print a clear notice when we can't
load the expected file, so the user isn't surprised when their ``LLM_BACKEND``
looks unset at runtime.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _fallback_parse(path: Path) -> int:
    """Minimal KEY=VALUE parser. Skips comments/blank lines; strips surrounding
    quotes. Does not overwrite existing environment variables."""
    loaded = 0
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val
            loaded += 1
    return loaded


def load_project_root_dotenv() -> None:
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if not env_path.is_file():
        return

    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_path)
        return
    except ImportError:
        n = _fallback_parse(env_path)
        print(
            f"[dotenv] python-dotenv not installed; used fallback parser to load {n} vars from {env_path}. "
            "`pip install python-dotenv` for canonical behavior.",
            file=sys.stderr,
        )
