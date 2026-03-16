"""
PCAB Task Registry: Data-driven task definitions for the Personalized Context Assembly Benchmark.

Tasks define required tools, gold schema for evaluation, and extraction parameters.
When running benchmarks with Mistral 7B routing, extraction params come from the task
definition rather than hard-coded keyword matching—enabling flexible evaluation.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractionParams:
    """Parameters for tool calls. Used when task config overrides keyword extraction."""

    contact_name: str | None = None
    date: str | None = None
    drive_query: str | None = None
    commute_origin: str | None = None
    commute_destination: str | None = None

    @property
    def commute_pair(self) -> tuple[str, str] | None:
        if self.commute_origin and self.commute_destination:
            return (self.commute_origin, self.commute_destination)
        return None

    def as_override_dict(self) -> dict[str, str]:
        """Export non-None fields for extraction_overrides."""
        d = {}
        if self.contact_name is not None:
            d["contact_name"] = self.contact_name
        if self.date is not None:
            d["date"] = self.date
        if self.drive_query is not None:
            d["drive_query"] = self.drive_query
        if self.commute_origin is not None:
            d["commute_origin"] = self.commute_origin
        if self.commute_destination is not None:
            d["commute_destination"] = self.commute_destination
        return d


@dataclass
class PCABTask:
    """A single PCAB benchmark task with evaluation metadata."""

    id: str
    description: str
    required_tools: list[str]  # e.g. ["get_contact_preferences", "get_calendar_events", "estimate_commute"]
    gold_schema: dict[str, Any]  # Expected output fields for strict binary success
    extraction_params: ExtractionParams = field(default_factory=ExtractionParams)
    category: str = "sequential"  # sequential | parallel | exploratory


# --- PCAB Benchmark Task Registry ---

PCAB_TASKS: list[PCABTask] = [
    PCABTask(
        id="PCAB-Seq-01",
        description="Find my advisor's next available slot, check conflicts, plan commute.",
        required_tools=[
            "get_contact_preferences",
            "get_calendar_events",
            "estimate_commute",
        ],
        gold_schema={
            "advisor_name": "Dr. Hong Man",
            "meeting_time": "11:00",
            "commute_time": 8,
        },
        extraction_params=ExtractionParams(
            contact_name="Dr. Hong Man",
            date="2026-03-16",
            commute_origin="Babbio Center",
            commute_destination="Gateway North",
        ),
        category="sequential",
    ),
    PCABTask(
        id="PCAB-Seq-02",
        description="I need to meet with Dr. Hong Man on March 16th after my Deep Learning lecture. Where should I go, and how long will it take to walk there?",
        required_tools=[
            "get_contact_preferences",
            "get_calendar_events",
            "estimate_commute",
        ],
        gold_schema={
            "advisor_name": "Dr. Hong Man",
            "location": "Burchard Hall",
            "commute_minutes": 8,
        },
        extraction_params=ExtractionParams(
            contact_name="Dr. Hong Man",
            date="2026-03-16",
            commute_origin="Babbio Center",
            commute_destination="Burchard Hall",
        ),
        category="sequential",
    ),
    PCABTask(
        id="PCAB-Par-01",
        description="Cross-reference my calendar availability with maps for upstate NY hiking and check my drive notes.",
        required_tools=[
            "get_calendar_events",
            "search_drive_docs",
            "estimate_commute",
        ],
        gold_schema={"sources_aggregated": 3},
        extraction_params=ExtractionParams(
            date="2026-03-16",
            drive_query="Capstone",
            commute_origin="Babbio Center",
            commute_destination="Gateway North",
        ),
        category="parallel",
    ),
    PCABTask(
        id="PCAB-Par-14",
        description="Please aggregate my Calendar and Maps data simultaneously.",
        required_tools=["get_calendar_events", "estimate_commute"],
        gold_schema={"sources_aggregated": 2},
        extraction_params=ExtractionParams(
            date="2026-03-16",
            commute_origin="Babbio Center",
            commute_destination="Gateway North",
        ),
        category="parallel",
    ),
    PCABTask(
        id="PCAB-Exploratory-03",
        description="Cross-reference 3 different data sources to find schedule anomalies.",
        required_tools=[
            "get_calendar_events",
            "search_drive_docs",
            "estimate_commute",
        ],
        gold_schema={"sources_checked": 3},
        extraction_params=ExtractionParams(
            date="2026-03-16",
            drive_query="notes",
            commute_origin="Gateway North",
            commute_destination="Library",
        ),
        category="exploratory",
    ),
]


def get_pcab_task(task_id: str) -> PCABTask | None:
    """Look up a PCAB task by ID."""
    for t in PCAB_TASKS:
        if t.id == task_id:
            return t
    return None


def get_pcab_tasks(category: str | None = None) -> list[PCABTask]:
    """Return all PCAB tasks, optionally filtered by category."""
    if category is None:
        return list(PCAB_TASKS)
    return [t for t in PCAB_TASKS if t.category == category]
