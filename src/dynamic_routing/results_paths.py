"""Helpers for organized timestamped benchmark output paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def benchmark_results_dir(project_root: Path, benchmark: str, run_kind: str) -> Path:
    return project_root / "results" / benchmark / run_kind


def timestamped_results_path(
    project_root: Path,
    benchmark: str,
    run_kind: str,
    user_path: Path,
    timestamp: str,
    *,
    default_suffix: str,
) -> Path:
    stem = user_path.stem
    suffix = user_path.suffix if user_path.suffix else default_suffix
    return benchmark_results_dir(project_root, benchmark, run_kind) / f"{stem}_{timestamp}{suffix}"


def workbench_run_kind(output_path: Path, limit: int) -> str:
    stem = output_path.stem.lower()
    if "fixture" in stem or "smoke" in stem or limit > 0:
        return "fixtures"
    return "sweeps"


def browsecomp_run_kind(dataset_mode: str) -> str:
    if dataset_mode == "fixture":
        return "fixtures"
    return "sweeps"


def finrate_run_kind(dataset_mode: str) -> str:
    if dataset_mode == "fixture":
        return "fixtures"
    return "converted"


def infer_benchmark_from_payload(data: dict[str, Any], source_path: Path) -> str:
    metadata = data.get("metadata") or {}
    phase = str(metadata.get("phase") or "").lower()
    name = source_path.name.lower()

    if "browsecomp" in phase or "browsecomp" in name:
        return "browsecomp"
    if "finrate" in phase or "finrate" in name or "fin-rate" in phase:
        return "finrate"
    if "workbench" in phase or "workbench" in name:
        return "workbench"
    if "pcab" in phase or "pcab" in name or source_path.name == "benchmark_results.json":
        return "pcab_legacy"
    return "misc"
