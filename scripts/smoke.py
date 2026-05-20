"""Manual end-to-end smoke test: parser → sampler → Gemini → audit.

Runs the full vidaudit pipeline on a real video against a real Gemini API
without going through the (not-yet-built) CLI. The goal is to surface plumbing
issues — wrong SDK call shape, schema mismatch, dud prompts, real-world spaCy
extraction noise — *before* we layer the report and CLI on top.

Not a unit test. Hits the live Gemini API; requires ``GEMINI_API_KEY``.
``argparse`` is used here rather than Typer on purpose so this script stays
independent of the real CLI (step 7), which CLAUDE.md §4 specifies uses Typer.

Usage:
    uv run python scripts/smoke.py \\
        --video path/to/clip.mp4 \\
        --descriptions path/to/descs.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from vidaudit.auditors.object_audit import SegmentAuditResult, audit_segment
from vidaudit.description_parser import parse_descriptions
from vidaudit.frame_sampler import sample_frames
from vidaudit.vlm.gemini import GeminiBackend

if TYPE_CHECKING:
    from pathlib import Path

    from vidaudit.description_parser import DescriptionSegment


_DEFAULT_CONTEXT_WINDOW = 1.0  # used only when timestamp_end is None

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
        "--verbose",
        action="store_true",
        help="Show INFO-level logs (rate-limit waits, cache hits, etc.).",
    )
    return parser.parse_args()


def _segment_sampling_plan(segment: DescriptionSegment) -> tuple[float, float]:
    """Pick a (primary_timestamp, context_window) for one segment.

    When ``timestamp_end`` is present we sample at the midpoint with the
    half-span as the window — primary lands in the middle, context lands at
    start and end (a coarse DD-9 span sampling that fits the current
    ``sample_frames`` signature). When the end is missing we fall back to
    ``(start, _DEFAULT_CONTEXT_WINDOW)``; the real DD-9 fallback chain
    (next-segment-start → ffprobe duration → cap) lives in the CLI step.
    """
    if segment.timestamp_end is None:
        return segment.timestamp_start, _DEFAULT_CONTEXT_WINDOW
    span = max(0.0, segment.timestamp_end - segment.timestamp_start)
    midpoint = segment.timestamp_start + span / 2
    half_span = span / 2 if span > 0 else _DEFAULT_CONTEXT_WINDOW
    return midpoint, half_span


def _render_segment(console: Console, audit: SegmentAuditResult) -> None:
    seg = audit.segment
    range_label = (
        f"{seg.timestamp_start:.1f}s"
        if seg.timestamp_end is None
        else f"{seg.timestamp_start:.1f}–{seg.timestamp_end:.1f}s"
    )
    verdict_text = Text(audit.verdict, style=_VERDICT_STYLE.get(audit.verdict, ""))

    header = Text.assemble(
        ("Segment ", "bold"),
        (range_label, "cyan"),
        ("  verdict=", "dim"),
        verdict_text,
        ("  grounding=", "dim"),
        (f"{audit.grounding_score:.2f}", "bold"),
        ("  flagged=", "dim"),
        (str(audit.hallucination_count), "bold red" if audit.hallucination_count else "bold"),
    )
    console.print(Panel.fit(seg.description, title=header, border_style="cyan"))

    if not audit.claim_results:
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
    for cr in audit.claim_results:
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


def _render_summary(console: Console, results: list[SegmentAuditResult]) -> None:
    total_claims = sum(len(r.claim_results) for r in results)
    total_flagged = sum(r.hallucination_count for r in results)
    supported = sum(
        1 for r in results for cr in r.claim_results if cr.verification.verdict == "supported"
    )
    aggregate_grounding = supported / total_claims if total_claims else 1.0

    summary = Table(title="Smoke summary", title_style="bold", show_header=False, box=None)
    summary.add_row("Segments audited", str(len(results)))
    summary.add_row("Total claims", str(total_claims))
    summary.add_row("Supported claims", str(supported))
    summary.add_row("Flagged (hallucination candidates)", str(total_flagged))
    summary.add_row("Aggregate grounding score", f"{aggregate_grounding:.3f}")
    console.print(summary)


def main() -> int:
    # Load .env (e.g. GEMINI_API_KEY) before anything reads os.environ.
    # No-op if no .env file is found.
    load_dotenv()

    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s — %(message)s",
    )

    console = Console()
    video_path: Path = _to_path(args.video, "video")
    descs_path: Path = _to_path(args.descriptions, "descriptions")

    console.rule("[bold]vidaudit smoke")
    console.print(f"video:        [cyan]{video_path}[/]")
    console.print(f"descriptions: [cyan]{descs_path}[/]")
    console.print(f"backend:      [cyan]{args.model}[/]")
    console.print()

    segments = parse_descriptions(descs_path)
    console.print(
        f"Parsed [bold]{len(segments)}[/] segments, "
        f"[bold]{sum(len(s.claims) for s in segments)}[/] claims total."
    )

    vlm = GeminiBackend(model=args.model, min_interval_seconds=args.min_interval)

    results: list[SegmentAuditResult] = []
    for segment in segments:
        if not segment.claims:
            console.print(
                f"[dim]Skipping segment at {segment.timestamp_start:.1f}s — "
                "no extractable claims.[/]"
            )
            continue

        primary_t, window = _segment_sampling_plan(segment)
        frames_by_t = sample_frames(video_path, [primary_t], context_window=window)
        frames = frames_by_t.get(primary_t)
        if not frames:
            console.print(
                f"[yellow]Skipping segment at {segment.timestamp_start:.1f}s — "
                "frame sampler returned no frames (outside duration?).[/]"
            )
            continue

        result = audit_segment(segment, frames, vlm)
        results.append(result)
        _render_segment(console, result)

    console.rule()
    _render_summary(console, results)
    return 0


def _to_path(raw: str, kind: str) -> Path:
    from pathlib import Path

    path = Path(raw).expanduser()
    if not path.exists():
        print(f"error: {kind} path does not exist: {path}", file=sys.stderr)
        raise SystemExit(2)
    return path


if __name__ == "__main__":
    sys.exit(main())
