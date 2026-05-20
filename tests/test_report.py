"""Tests for vidaudit.report."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import cast

import pytest
from rich.console import Console

from vidaudit.auditors.object_audit import (
    ClaimResult,
    SegmentAuditResult,
    SegmentVerdict,
)
from vidaudit.description_parser import Claim, DescriptionSegment
from vidaudit.report import (
    AuditReport,
    build_report,
    render_terminal,
)
from vidaudit.vlm.base import Verdict, VerificationResult

# ---- helpers --------------------------------------------------------------


def _audit(
    text: str,
    verdict: str,
    confidence: float,
    flagged: bool,
    *,
    seg_verdict: str | None = None,
    ts_end: float | None = 5.0,
) -> SegmentAuditResult:
    """One-claim SegmentAuditResult convenience constructor."""
    claim = Claim(text=text, claim_type="object", source_description="ignored")
    verif = VerificationResult(
        claim=text,
        verdict=cast("Verdict", verdict),
        confidence=confidence,
        evidence="",
    )
    cr = ClaimResult(claim=claim, verification=verif, flagged=flagged)

    if seg_verdict is None:
        seg_verdict = "clean" if verdict == "supported" else "full_hallucination"

    return SegmentAuditResult(
        segment=DescriptionSegment(
            timestamp_start=0.0,
            timestamp_end=ts_end,
            description="ignored",
            claims=[claim],
        ),
        claim_results=[cr],
        grounding_score=1.0 if verdict == "supported" else 0.0,
        hallucination_count=1 if flagged else 0,
        verdict=cast("SegmentVerdict", seg_verdict),
    )


def _kwargs() -> dict[str, object]:
    return {
        "video_path": Path("video.mp4"),
        "backend_id": "gemini-2.5-flash",
        "confidence_threshold": 0.3,
        "clean_threshold": 0.8,
        "partial_threshold": 0.4,
    }


# ---- tests ----------------------------------------------------------------


def test_summary_counts_verified_flagged_and_uncertain() -> None:
    a_supported = _audit("woman", "supported", 0.9, flagged=False)
    a_flagged = _audit("unicorn", "unsupported", 0.9, flagged=True)
    a_uncertain = _audit(
        "maybe", "uncertain", 0.4, flagged=False, seg_verdict="partial_hallucination"
    )

    report = build_report(
        [(a_supported, False), (a_flagged, False), (a_uncertain, True)],
        **_kwargs(),  # type: ignore[arg-type]
    )

    assert report.summary.total_descriptions == 3
    assert report.summary.total_claims == 3
    assert report.summary.verified_claims == 1
    assert report.summary.hallucinated_claims == 1
    assert report.summary.uncertain_claims == 1
    assert report.summary.overall_grounding_score == pytest.approx(1 / 3)
    # 2 of 3 are not "clean" (a_flagged is full_hallucination, a_uncertain is partial)
    assert report.summary.descriptions_flagged == 2


def test_report_round_trips_through_json() -> None:
    audit = _audit("woman", "supported", 0.9, flagged=False)
    report = build_report([(audit, False)], **_kwargs())  # type: ignore[arg-type]

    restored = AuditReport.model_validate_json(report.model_dump_json())

    assert restored == report


def test_save_json_writes_valid_output(tmp_path: Path) -> None:
    audit = _audit("woman", "supported", 0.9, flagged=False)
    report = build_report([(audit, False)], **_kwargs())  # type: ignore[arg-type]
    path = tmp_path / "report.json"

    report.save_json(path)

    data = json.loads(path.read_text())
    assert data["metadata"]["backend"] == "gemini-2.5-flash"
    assert data["metadata"]["vidaudit_version"] == "0.1.0"
    assert data["metadata"]["confidence_threshold"] == 0.3
    assert data["summary"]["total_claims"] == 1
    assert len(data["segments"]) == 1
    assert data["segments"][0]["end_inferred"] is False


def test_save_json_creates_missing_parent_directories(tmp_path: Path) -> None:
    audit = _audit("x", "supported", 0.9, flagged=False)
    report = build_report([(audit, False)], **_kwargs())  # type: ignore[arg-type]
    path = tmp_path / "nested" / "subdir" / "report.json"

    report.save_json(path)

    assert path.exists()


def test_end_inferred_is_surfaced_per_segment() -> None:
    a1 = _audit("woman", "supported", 0.9, flagged=False)
    a2 = _audit("dog", "supported", 0.9, flagged=False)

    report = build_report(
        [(a1, False), (a2, True)],  # second segment had its end inferred
        **_kwargs(),  # type: ignore[arg-type]
    )

    assert report.segments[0].end_inferred is False
    assert report.segments[1].end_inferred is True


def test_empty_audit_is_vacuously_clean() -> None:
    report = build_report([], **_kwargs())  # type: ignore[arg-type]

    assert report.summary.total_descriptions == 0
    assert report.summary.total_claims == 0
    assert report.summary.overall_grounding_score == 1.0
    assert report.summary.descriptions_flagged == 0
    assert report.segments == []


def test_render_terminal_runs_without_crashing() -> None:
    a_supported = _audit("woman", "supported", 0.9, flagged=False)
    a_flagged = _audit("unicorn", "unsupported", 0.9, flagged=True)
    report = build_report(
        [(a_supported, False), (a_flagged, True)],
        **_kwargs(),  # type: ignore[arg-type]
    )

    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=200)
    render_terminal(report, console)

    output = buf.getvalue()
    assert "vidaudit report" in output
    assert "woman" in output
    assert "unicorn" in output
    assert "Summary" in output
