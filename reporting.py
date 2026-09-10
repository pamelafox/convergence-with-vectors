"""Result rows, CSV/JSON writers, and terminal tables for the experiments.

All three CLIs (``compare_pair.py``, ``compare_batch.py``,
``simulate_convergence.py``) funnel their results through :class:`ResultRow`
so the machine-readable CSV/JSON output has one consistent schema regardless
of which script produced it.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from operators import CandidateScore

#: Column order used for both CSV and JSON output.
FIELDNAMES: list[str] = [
    "word_a",
    "word_b",
    "model",
    "operator",
    "template",
    "round",
    "rank",
    "candidate",
    "score",
    "sim_a",
    "sim_b",
    "outcome",
]


@dataclass
class ResultRow:
    """One ranked candidate, tagged with everything needed to interpret it."""

    word_a: str
    word_b: str
    model: str
    operator: str
    template: str | None
    round: int | None
    rank: int
    candidate: str
    score: float
    sim_a: float
    sim_b: float
    outcome: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def candidate_rows(
    word_a: str,
    word_b: str,
    model: str,
    operator: str,
    candidates: list[CandidateScore],
    *,
    template: str | None = None,
    round_number: int | None = None,
    outcome: str | None = None,
) -> list[ResultRow]:
    """Convert a list of :class:`~operators.CandidateScore` into report rows."""
    return [
        ResultRow(
            word_a=word_a,
            word_b=word_b,
            model=model,
            operator=operator,
            template=template,
            round=round_number,
            rank=c.rank,
            candidate=c.candidate,
            score=c.score,
            sim_a=c.sim_a,
            sim_b=c.sim_b,
            outcome=outcome,
        )
        for c in candidates
    ]


def write_csv(rows: list[ResultRow], path: str | Path) -> None:
    """Write rows as CSV using the shared :data:`FIELDNAMES` column order."""
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            record = {k: ("" if v is None else v) for k, v in row.as_dict().items()}
            writer.writerow(record)


def write_json(rows: list[ResultRow], path: str | Path) -> None:
    """Write rows as a JSON array of objects."""
    Path(path).write_text(json.dumps([r.as_dict() for r in rows], indent=2), encoding="utf-8")


def render_side_by_side(
    rows_by_model: dict[str, list[ResultRow]],
    console: Console | None = None,
    title: str = "",
) -> None:
    """Print one compact table per model, laid out side by side for comparison."""
    console = console or Console()
    if title:
        console.rule(title)
    tables = []
    for model, rows in rows_by_model.items():
        table = Table(title=model)
        for col in ("Rank", "Candidate", "Score", "Sim A", "Sim B"):
            table.add_column(col)
        for row in rows:
            table.add_row(str(row.rank), row.candidate, f"{row.score:.4f}", f"{row.sim_a:.4f}", f"{row.sim_b:.4f}")
        tables.append(table)
    console.print(Columns(tables, equal=True, expand=True))


def render_flat_table(rows: list[ResultRow], console: Console | None = None, title: str = "") -> None:
    """Print a single flat table (used for batch/convergence summaries)."""
    console = console or Console()
    table = Table(title=title)
    for col in ("Word A", "Word B", "Model", "Operator", "Template", "Round", "Rank", "Candidate", "Score", "Outcome"):
        table.add_column(col)
    for row in rows:
        table.add_row(
            row.word_a,
            row.word_b,
            row.model,
            row.operator,
            row.template or "",
            "" if row.round is None else str(row.round),
            str(row.rank),
            row.candidate,
            f"{row.score:.4f}",
            row.outcome or "",
        )
    console.print(table)
