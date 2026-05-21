"""Manual end-to-end smoke test: parser → sampler → Gemini → audit.

Runs the full vidaudit pipeline on a real video against a real Gemini API
without going through the (not-yet-built) CLI. The goal is to surface plumbing
issues — wrong SDK call shape, schema mismatch, dud prompts, real-world spaCy
extraction noise — *before* we layer the CLI on top.

Not a unit test. Hits the live Gemini API; requires ``GEMINI_API_KEY``.
``argparse`` is used here rather than Typer on purpose so this script stays
independent of the real CLI (step 7), which CLAUDE.md §4 specifies uses Typer.

Usage:
    uv run python scripts/smoke.py \\
        --video path/to/clip.mp4 \\
        --descriptions path/to/descs.json [--output report.json]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from rich.console import Console

from vidaudit.auditors.object_audit import SegmentAuditResult, audit_segment
from vidaudit.description_parser import parse_descriptions
from vidaudit.frame_sampler import sample_frames
from vidaudit.report import build_report, render_terminal
from vidaudit.vlm.gemini import GeminiBackend

if TYPE_CHECKING:
    from vidaudit.description_parser import DescriptionSegment


_DEFAULT_CONTEXT_WINDOW = 1.0  # used only when timestamp_end is None
_DEFAULT_CONFIDENCE_THRESHOLD = 0.3
_DEFAULT_CLEAN_THRESHOLD = 0.8
_DEFAULT_PARTIAL_THRESHOLD = 0.4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--video", required=True, type=str, help="Path to the video file.")
    parser.add_argument(
        "--descriptions",
        required=True,
        type=str,
        help="Path to the descriptions JSON file.",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model id (default: gemini-2.5-flash).",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=4.0,
        help="Seconds between Gemini calls for free-tier pacing (default: 4.0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the audit report as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show INFO-level logs (rate-limit waits, cache hits, etc.).",
    )
    return parser.parse_args()


def _segment_sampling_plan(segment: DescriptionSegment) -> tuple[float, float]:
    """Pick a (primary_timestamp, context_window) for one segment.

    When ``timestamp_end`` is present we sample at the midpoint with the
    half-span as the window — primary lands in the middle, context lands at
    start and end. When the end is missing we fall back to
    ``(start, _DEFAULT_CONTEXT_WINDOW)``; the full missing-end resolution
    chain (next-segment-start → ffprobe duration → cap) lives in the CLI.
    """
    if segment.timestamp_end is None:
        return segment.timestamp_start, _DEFAULT_CONTEXT_WINDOW
    span = max(0.0, segment.timestamp_end - segment.timestamp_start)
    midpoint = segment.timestamp_start + span / 2
    half_span = span / 2 if span > 0 else _DEFAULT_CONTEXT_WINDOW
    return midpoint, half_span


def main() -> int:
    # Load .env (e.g. GEMINI_API_KEY) before anything reads os.environ.
    load_dotenv()

    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s — %(message)s",
    )

    console = Console()
    video_path = _existing_path(args.video, "video")
    descs_path = _existing_path(args.descriptions, "descriptions")

    console.print(f"Loading descriptions from [cyan]{descs_path}[/]")
    segments = parse_descriptions(descs_path)
    console.print(
        f"Parsed [bold]{len(segments)}[/] segments, "
        f"[bold]{sum(len(s.claims) for s in segments)}[/] claims total.\n"
    )

    vlm = GeminiBackend(model=args.model, min_interval_seconds=args.min_interval)

    audited: list[tuple[SegmentAuditResult, bool]] = []
    for segment in segments:
        if not segment.claims:
            console.print(
                f"[dim]Skipping segment at {segment.timestamp_start:.1f}s — "
                "no extractable claims.[/]"
            )
            continue

        primary_t, window = _segment_sampling_plan(segment)
        console.print(
            f"[dim]Auditing {segment.timestamp_start:.1f}s ({len(segment.claims)} claims)…[/]"
        )
        frames_by_t = sample_frames(video_path, [primary_t], context_window=window)
        frames = frames_by_t.get(primary_t)
        if not frames:
            console.print(
                "[yellow]Skipped — frame sampler returned no frames (outside duration?).[/]"
            )
            continue

        audit = audit_segment(
            segment,
            frames,
            vlm,
            confidence_threshold=_DEFAULT_CONFIDENCE_THRESHOLD,
            clean_threshold=_DEFAULT_CLEAN_THRESHOLD,
            partial_threshold=_DEFAULT_PARTIAL_THRESHOLD,
        )
        # Smoke does not run the full missing-end resolution chain — any
        # timestamp_end already in the JSON is taken as-is, never inferred.
        audited.append((audit, False))

    report = build_report(
        audited,
        video_path=video_path,
        backend_id=vlm.model_id,
        confidence_threshold=_DEFAULT_CONFIDENCE_THRESHOLD,
        clean_threshold=_DEFAULT_CLEAN_THRESHOLD,
        partial_threshold=_DEFAULT_PARTIAL_THRESHOLD,
    )

    render_terminal(report, console)

    if args.output:
        out_path = Path(args.output).expanduser()
        report.save_json(out_path)
        console.print(f"\n[dim]Report written to {out_path}[/]")

    return 0


def _existing_path(raw: str, kind: str) -> Path:
    path = Path(raw).expanduser()
    if not path.exists():
        print(f"error: {kind} path does not exist: {path}", file=sys.stderr)
        raise SystemExit(2)
    return path


if __name__ == "__main__":
    sys.exit(main())
