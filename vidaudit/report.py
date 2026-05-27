"""Aggregate audit results into a structured report (JSON + Rich terminal).

The report is the project's deliverable shape — it carries enough metadata
(model id, thresholds, version, per-segment ``end_inferred`` flags) to be
reproducible and self-explanatory. ``end_inferred`` records whether the
orchestration filled in a missing ``timestamp_end`` — an inferred end is never
silently fabricated, it is surfaced in the report.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from vidaudit import __version__
from vidaudit.auditors.object_audit import (
    ClaimResult,
    SegmentAuditResult,
    SegmentVerdict,
)

if TYPE_CHECKING:
    from pathlib import Path


_VERDICT_STYLE = {
    "clean": "bold green",
    "partial_hallucination": "bold yellow",
    "full_hallucination": "bold red",
}

_CLAIM_STYLE = {
    "supported": "green",
    "unsupported": "red",
    "uncertain": "yellow",
}


class ReportMetadata(BaseModel):
    """Metadata sufficient to reproduce the audit run."""

    video: str
    backend: str  # ``VLMBackend.model_id`` (a pinned, reproducible identifier)
    generated_at: datetime
    confidence_threshold: float
    clean_threshold: float
    partial_threshold: float
    vidaudit_version: str


class ReportSummary(BaseModel):
    """Aggregate counts across all audited segments."""

    total_descriptions: int
    total_claims: int
    verified_claims: int  # verification.verdict == "supported"
    hallucinated_claims: int  # ClaimResult.flagged is True
    uncertain_claims: int  # neither supported nor flagged (incl. low-confidence)
    overall_grounding_score: float  # verified / total
    descriptions_flagged: int  # segments with verdict != "clean"


class ReportSegment(BaseModel):
    """One segment in the report. Flat shape — easier to consume than nested."""

    timestamp_start: float
    timestamp_end: float | None
    end_inferred: bool = False  # True iff orchestration filled a missing end
    description: str
    grounding_score: float
    hallucination_count: int
    verdict: SegmentVerdict
    claims: list[ClaimResult]

    @classmethod
    def from_audit(cls, audit: SegmentAuditResult, *, end_inferred: bool = False) -> ReportSegment:
        return cls(
            timestamp_start=audit.segment.timestamp_start,
            timestamp_end=audit.segment.timestamp_end,
            end_inferred=end_inferred,
            description=audit.segment.description,
            grounding_score=audit.grounding_score,
            hallucination_count=audit.hallucination_count,
            verdict=audit.verdict,
            claims=audit.claim_results,
        )


class AuditReport(BaseModel):
    """The full audit report — root model written to disk and printed."""

    metadata: ReportMetadata
    summary: ReportSummary
    segments: list[ReportSegment]

    def save_json(self, path: Path, *, indent: int = 2) -> None:
        """Write the report as pretty-printed JSON to ``path``.

        Creates parent directories as needed.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=indent))


def build_report(
    audited_segments: list[tuple[SegmentAuditResult, bool]],
    *,
    video_path: Path,
    backend_id: str,
    confidence_threshold: float,
    clean_threshold: float,
    partial_threshold: float,
) -> AuditReport:
    """Aggregate audited segments into the final :class:`AuditReport`.

    Args:
        audited_segments: ``(audit, end_inferred)`` for each audited segment.
            ``end_inferred`` is ``True`` iff the orchestration filled a
            missing ``timestamp_end`` — surfaced in the report so the
            inference is never silent.
        video_path: Path to the source video (recorded in metadata).
        backend_id: ``VLMBackend.model_id``, e.g. ``gemini-2.5-flash``.
        confidence_threshold: Verdict-confidence threshold the audit ran with.
        clean_threshold: Grounding-score cutoff used for the verdict.
        partial_threshold: Grounding-score cutoff used for the verdict.

    Returns:
        The complete :class:`AuditReport`.
    """
    segments = [
        ReportSegment.from_audit(audit, end_inferred=inferred)
        for audit, inferred in audited_segments
    ]

    total_descriptions = len(segments)
    total_claims = sum(len(s.claims) for s in segments)
    verified = sum(1 for s in segments for c in s.claims if c.verification.verdict == "supported")
    flagged = sum(s.hallucination_count for s in segments)
    uncertain = total_claims - verified - flagged
    descriptions_flagged = sum(1 for s in segments if s.verdict != "clean")
    overall_grounding = verified / total_claims if total_claims else 1.0

    return AuditReport(
        metadata=ReportMetadata(
            video=str(video_path),
            backend=backend_id,
            generated_at=datetime.now(timezone.utc),
            confidence_threshold=confidence_threshold,
            clean_threshold=clean_threshold,
            partial_threshold=partial_threshold,
            vidaudit_version=__version__,
        ),
        summary=ReportSummary(
            total_descriptions=total_descriptions,
            total_claims=total_claims,
            verified_claims=verified,
            hallucinated_claims=flagged,
            uncertain_claims=uncertain,
            overall_grounding_score=overall_grounding,
            descriptions_flagged=descriptions_flagged,
        ),
        segments=segments,
    )


def render_terminal(report: AuditReport, console: Console | None = None) -> None:
    """Pretty-print the report to a Rich console."""
    if console is None:
        console = Console()

    console.rule("[bold]vidaudit report")
    console.print(
        f"video:        [cyan]{report.metadata.video}[/]\n"
        f"backend:      [cyan]{report.metadata.backend}[/]\n"
        f"generated:    [cyan]{report.metadata.generated_at.isoformat()}[/]"
    )
    console.print()

    for seg in report.segments:
        _render_segment(console, seg)

    console.rule()
    _render_summary(console, report)


def _render_segment(console: Console, seg: ReportSegment) -> None:
    if seg.timestamp_end is None:
        range_label = f"{seg.timestamp_start:.1f}s"
    else:
        range_label = f"{seg.timestamp_start:.1f}–{seg.timestamp_end:.1f}s"
    if seg.end_inferred:
        range_label += " (end inferred)"

    verdict_text = Text(seg.verdict, style=_VERDICT_STYLE.get(seg.verdict, ""))
    header = Text.assemble(
        ("Segment ", "bold"),
        (range_label, "cyan"),
        ("  verdict=", "dim"),
        verdict_text,
        ("  grounding=", "dim"),
        (f"{seg.grounding_score:.2f}", "bold"),
        ("  flagged=", "dim"),
        (str(seg.hallucination_count), "bold red" if seg.hallucination_count else "bold"),
    )
    console.print(Panel.fit(seg.description, title=header, border_style="cyan"))

    if not seg.claims:
        console.print("  [dim italic]no claims extracted[/]")
        return

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("", width=2)
    table.add_column("Claim", style="bold")
    table.add_column("Type", style="dim")
    table.add_column("Verdict")
    table.add_column("Conf", justify="right")
    table.add_column("Flag", justify="center")
    table.add_column("Evidence", overflow="fold", max_width=60)
    for cr in seg.claims:
        verdict = cr.verification.verdict
        marker = "✗" if cr.flagged else ("✓" if verdict == "supported" else "·")
        marker_style = "red" if cr.flagged else ("green" if verdict == "supported" else "yellow")
        table.add_row(
            Text(marker, style=marker_style),
            cr.claim.text,
            cr.claim.claim_type,
            Text(verdict, style=_CLAIM_STYLE.get(verdict, "")),
            f"{cr.verification.confidence:.2f}",
            Text("FLAG", style="bold red") if cr.flagged else Text("·", style="dim"),
            cr.verification.evidence,
        )
    console.print(table)
    console.print()


def _render_summary(console: Console, report: AuditReport) -> None:
    s = report.summary
    table = Table(title="Summary", title_style="bold", show_header=False, box=None)
    table.add_row("Segments audited", str(s.total_descriptions))
    table.add_row("Total claims", str(s.total_claims))
    table.add_row("Verified (supported)", str(s.verified_claims))
    table.add_row("Flagged (hallucinated)", str(s.hallucinated_claims))
    table.add_row("Uncertain", str(s.uncertain_claims))
    table.add_row("Overall grounding score", f"{s.overall_grounding_score:.3f}")
    table.add_row("Segments flagged", str(s.descriptions_flagged))
    console.print(table)
