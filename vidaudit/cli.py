"""Typer CLI for vidaudit.

Two subcommands:

* ``vidaudit audit ...`` — full pipeline: parse → sample → verify → report
* ``vidaudit parse ...`` — claim extraction only, no VLM calls (debug)

The CLI is a thin wrapper around the orchestration pipeline. The missing-end
resolution chain (next-segment start → ffprobe duration → cap → point
fallback) lives here because it needs both the ordered segment list and the
video, neither of which the auditor sees.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from vidaudit.auditors.object_audit import SegmentAuditResult, audit_segment
from vidaudit.description_parser import parse_descriptions
from vidaudit.frame_sampler import get_video_duration, sample_frames
from vidaudit.report import AuditReport, build_report, render_terminal
from vidaudit.vlm.gemini import GeminiBackend
from vidaudit.vlm.qwen_vl import QwenVLBackend, model_from_env

if TYPE_CHECKING:
    from collections.abc import Callable

    from vidaudit.description_parser import DescriptionSegment
    from vidaudit.vlm.base import VLMBackend


_DEFAULT_FALLBACK_CONTEXT_WINDOW = 1.0


class Backend(str, Enum):
    gemini = "gemini"
    qwen = "qwen"  # implemented in a later step


app = typer.Typer(
    name="vidaudit",
    help="Audit VLM-generated video descriptions for hallucinations.",
    add_completion=False,
    no_args_is_help=True,
)


def resolve_segment_plan(
    segment: DescriptionSegment,
    *,
    next_start: float | None,
    video_duration: Callable[[], float],
    max_segment_span: float,
    fallback_context_window: float = _DEFAULT_FALLBACK_CONTEXT_WINDOW,
) -> tuple[float, float, bool]:
    """Resolve a single segment's sampling plan.

    Returns ``(primary_timestamp, context_window, end_inferred)``. Implements
    the missing-end resolution chain: when ``timestamp_end`` is ``None``,
    fall back to the next segment's ``timestamp_start``, else probe the
    video duration. Cap the resolved span at ``max_segment_span`` so a lone
    timestamp with a large trailing gap doesn't span forever. A degenerate
    span (≤ 0) collapses to point sampling with ``fallback_context_window``.

    ``video_duration`` is a callable so the probe runs only when actually
    needed (the last segment whose end is missing).
    """
    inferred = segment.timestamp_end is None
    end = segment.timestamp_end
    if end is None:
        end = next_start if next_start is not None else video_duration()

    span = end - segment.timestamp_start
    if span > max_segment_span:
        span = max_segment_span

    if span <= 0:
        return segment.timestamp_start, fallback_context_window, inferred

    midpoint = segment.timestamp_start + span / 2
    return midpoint, span / 2, inferred


def run_audit_pipeline(
    *,
    video_path: Path,
    descriptions_path: Path,
    vlm: VLMBackend,
    confidence_threshold: float,
    clean_threshold: float,
    partial_threshold: float,
    max_segment_span: float,
    fallback_context_window: float = _DEFAULT_FALLBACK_CONTEXT_WINDOW,
    console: Console | None = None,
) -> AuditReport:
    """Run the full audit pipeline and return the report.

    Caller owns I/O (writing JSON, printing the report). Factored out from
    the Typer command so tests can drive it directly with a fake backend.
    """
    if console is None:
        console = Console()

    console.print(f"Parsing [cyan]{descriptions_path}[/]…")
    segments = parse_descriptions(descriptions_path)
    console.print(
        f"Parsed [bold]{len(segments)}[/] segments, "
        f"[bold]{sum(len(s.claims) for s in segments)}[/] claims.\n"
    )

    # Probe video duration lazily — only if a missing-end segment falls last
    # in the list and we have no next-segment fallback.
    duration_cache: list[float] = []

    def _video_duration() -> float:
        if not duration_cache:
            duration_cache.append(get_video_duration(video_path))
        return duration_cache[0]

    plans = [
        resolve_segment_plan(
            seg,
            next_start=segments[i + 1].timestamp_start if i + 1 < len(segments) else None,
            video_duration=_video_duration,
            max_segment_span=max_segment_span,
            fallback_context_window=fallback_context_window,
        )
        for i, seg in enumerate(segments)
    ]

    audited: list[tuple[SegmentAuditResult, bool]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("Auditing", total=len(segments))
        for segment, (primary_t, window, end_inferred) in zip(segments, plans, strict=True):
            progress.update(task_id, description=f"Auditing {segment.timestamp_start:.1f}s")
            if not segment.claims:
                progress.advance(task_id)
                continue

            frames_by_t = sample_frames(video_path, [primary_t], context_window=window)
            frames = frames_by_t.get(primary_t)
            if frames is None:
                progress.advance(task_id)
                continue

            audit = audit_segment(
                segment,
                frames,
                vlm,
                confidence_threshold=confidence_threshold,
                clean_threshold=clean_threshold,
                partial_threshold=partial_threshold,
            )
            audited.append((audit, end_inferred))
            progress.advance(task_id)

    return build_report(
        audited,
        video_path=video_path,
        backend_id=vlm.model_id,
        confidence_threshold=confidence_threshold,
        clean_threshold=clean_threshold,
        partial_threshold=partial_threshold,
    )


@app.command()
def audit(
    video: Path = typer.Option(
        ...,
        "--video",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the video file.",
    ),
    descriptions: Path = typer.Option(
        ...,
        "--descriptions",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the descriptions JSON file.",
    ),
    output: Path = typer.Option(
        Path("report.json"),
        "--output",
        help="Where to write the audit report JSON.",
    ),
    backend: Backend = typer.Option(
        Backend.gemini,
        "--backend",
        help="VLM backend to use.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Override the backend's default model id.",
    ),
    confidence_threshold: float = typer.Option(
        0.3,
        "--confidence-threshold",
        min=0.0,
        max=1.0,
        help="Verdict-confidence threshold for flagging/rescue.",
    ),
    clean_threshold: float = typer.Option(
        0.8,
        "--clean-threshold",
        min=0.0,
        max=1.0,
        help="Grounding-score cutoff for the 'clean' verdict.",
    ),
    partial_threshold: float = typer.Option(
        0.4,
        "--partial-threshold",
        min=0.0,
        max=1.0,
        help="Grounding-score cutoff for 'partial_hallucination'.",
    ),
    max_segment_span: float = typer.Option(
        30.0,
        "--max-segment-span",
        help="Cap on inferred-end segment span in seconds.",
    ),
    min_interval: float = typer.Option(
        4.0,
        "--min-interval",
        help="Seconds between API calls (Gemini free-tier pacing).",
    ),
    qwen_revision: str | None = typer.Option(
        None,
        "--qwen-revision",
        help="Pin a Qwen model commit SHA for reproducible eval runs.",
    ),
    qwen_4bit: bool = typer.Option(
        False,
        "--qwen-4bit",
        help="Load the Qwen model in 4-bit (requires bitsandbytes).",
    ),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """Run the full audit pipeline and write a report."""
    load_dotenv()
    _setup_logging(verbose)

    console = Console()
    vlm = _build_backend(
        backend,
        model=model,
        min_interval=min_interval,
        qwen_revision=qwen_revision,
        qwen_4bit=qwen_4bit,
    )

    report = run_audit_pipeline(
        video_path=video,
        descriptions_path=descriptions,
        vlm=vlm,
        confidence_threshold=confidence_threshold,
        clean_threshold=clean_threshold,
        partial_threshold=partial_threshold,
        max_segment_span=max_segment_span,
        console=console,
    )

    render_terminal(report, console)
    report.save_json(output)
    console.print(f"\nReport written to [cyan]{output}[/]")


@app.command(name="parse")
def parse_cmd(
    descriptions: Path = typer.Option(
        ...,
        "--descriptions",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the descriptions JSON file.",
    ),
) -> None:
    """Extract and print claims without running VLM verification."""
    segments = parse_descriptions(descriptions)
    console = Console()
    total = 0
    for seg in segments:
        end_label = f"{seg.timestamp_end:.1f}" if seg.timestamp_end is not None else "?"
        console.print(f"[bold cyan]{seg.timestamp_start:.1f}–{end_label}s[/] {seg.description}")
        if not seg.claims:
            console.print("  [dim italic]no claims extracted[/]")
        for claim in seg.claims:
            console.print(f"  [dim]{claim.claim_type:>9}[/] {claim.text}")
        console.print()
        total += len(seg.claims)
    console.print(f"[bold]{len(segments)}[/] segments, [bold]{total}[/] claims total.")


def _build_backend(
    backend: Backend,
    *,
    model: str | None,
    min_interval: float,
    qwen_revision: str | None = None,
    qwen_4bit: bool = False,
) -> VLMBackend:
    if backend is Backend.gemini:
        if model is not None:
            return GeminiBackend(model=model, min_interval_seconds=min_interval)
        return GeminiBackend(min_interval_seconds=min_interval)
    if backend is Backend.qwen:
        # ``model`` defaults to None — fall through to the VIDAUDIT_QWEN_MODEL
        # env override (e.g. a 7B variant) then the qwen_vl default.
        return QwenVLBackend(
            model=model or model_from_env(),
            revision=qwen_revision,
            load_in_4bit=qwen_4bit,
        )
    raise typer.BadParameter(f"Unknown backend: {backend}")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s — %(message)s",
    )


if __name__ == "__main__":
    app()
