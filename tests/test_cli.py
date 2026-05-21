"""Tests for the Typer CLI.

The pipeline integration tests mock both ``sample_frames`` and the VLM —
no ffmpeg, no Gemini API. ``resolve_segment_plan`` is tested as a pure
function (no I/O at all).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest
from PIL import Image
from typer.testing import CliRunner

from vidaudit import cli
from vidaudit.cli import app, resolve_segment_plan, run_audit_pipeline
from vidaudit.description_parser import Claim, DescriptionSegment
from vidaudit.vlm.base import Verdict, VerificationResult, VLMBackend

if TYPE_CHECKING:
    from pathlib import Path

# ---- helpers --------------------------------------------------------------


def _segment(
    start: float, end: float | None, *claims: str, description: str = "x"
) -> DescriptionSegment:
    return DescriptionSegment(
        timestamp_start=start,
        timestamp_end=end,
        description=description,
        claims=[
            Claim(text=c, claim_type="object", source_description=description) for c in claims
        ],
    )


class _FakeVLM(VLMBackend):
    """Always-supported VLM for pipeline integration tests."""

    def __init__(self) -> None:
        self.model_id = "fake/1"

    def verify_claim(self, image: Image.Image, claim: str) -> VerificationResult:
        return VerificationResult(
            claim=claim,
            verdict=cast("Verdict", "supported"),
            confidence=0.9,
            evidence="ok",
        )


# ---- resolve_segment_plan ------------------------------------------------


def _never() -> float:
    raise AssertionError("video_duration() should not have been called")


def test_plan_uses_explicit_end_when_present() -> None:
    seg = _segment(10.0, 14.0)

    primary, window, inferred = resolve_segment_plan(
        seg,
        next_start=20.0,  # ignored when end is present
        video_duration=_never,
        max_segment_span=30.0,
    )

    assert primary == 12.0  # midpoint of [10, 14]
    assert window == 2.0
    assert inferred is False


def test_plan_falls_back_to_next_segment_start_when_end_missing() -> None:
    seg = _segment(10.0, None)

    primary, window, inferred = resolve_segment_plan(
        seg,
        next_start=20.0,
        video_duration=_never,  # next-start takes priority over duration
        max_segment_span=30.0,
    )

    assert primary == 15.0  # midpoint of [10, 20]
    assert window == 5.0
    assert inferred is True


def test_plan_falls_back_to_video_duration_for_last_segment() -> None:
    seg = _segment(40.0, None)
    duration_calls: list[int] = []

    def duration() -> float:
        duration_calls.append(1)
        return 50.0

    primary, window, inferred = resolve_segment_plan(
        seg,
        next_start=None,
        video_duration=duration,
        max_segment_span=30.0,
    )

    assert primary == 45.0  # midpoint of [40, 50]
    assert window == 5.0
    assert inferred is True
    assert len(duration_calls) == 1  # probed exactly once


def test_plan_caps_span_at_max_segment_span() -> None:
    seg = _segment(10.0, None)

    primary, window, inferred = resolve_segment_plan(
        seg,
        next_start=200.0,
        video_duration=_never,
        max_segment_span=30.0,
    )

    # Span capped to 30 → midpoint = 25, half-span = 15
    assert primary == 25.0
    assert window == 15.0
    assert inferred is True


def test_plan_falls_back_to_point_sampling_on_degenerate_span() -> None:
    seg = _segment(10.0, None)

    primary, window, inferred = resolve_segment_plan(
        seg,
        next_start=10.0,  # span collapses to 0
        video_duration=_never,
        max_segment_span=30.0,
        fallback_context_window=1.5,
    )

    assert primary == 10.0  # back to the point
    assert window == 1.5  # fallback window
    assert inferred is True


# ---- run_audit_pipeline integration ---------------------------------------


@pytest.fixture
def descriptions_file(tmp_path: Path) -> Path:
    path = tmp_path / "descs.json"
    path.write_text(
        json.dumps(
            [
                {
                    "timestamp_start": 1.0,
                    "timestamp_end": 4.0,
                    "description": "A dog runs across the grass",
                },
                {
                    "timestamp_start": 5.0,
                    "timestamp_end": None,  # triggers next-start fallback
                    "description": "A child laughs in the sunshine",
                },
            ]
        )
    )
    return path


def test_pipeline_runs_end_to_end_with_fake_backend(
    descriptions_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """parser → fake sampler → fake VLM → report."""

    monkeypatch.setenv("VIDAUDIT_CACHE_DIR", str(tmp_path / "cache"))
    fake_video = tmp_path / "v.mp4"
    fake_video.write_bytes(b"not-a-video")

    def fake_sample_frames(
        _path: Path, timestamps: list[float], context_window: float = 1.0
    ) -> dict[float, list[Image.Image]]:
        img = Image.new("RGB", (4, 4), color="red")
        return {t: [img, img, img] for t in timestamps}

    monkeypatch.setattr(cli, "sample_frames", fake_sample_frames)
    # Second segment has no end and is last → orchestration falls back to
    # video duration. Stub the probe so we don't shell out to ffprobe.
    monkeypatch.setattr(cli, "get_video_duration", lambda _p: 50.0)

    report = run_audit_pipeline(
        video_path=fake_video,
        descriptions_path=descriptions_file,
        vlm=_FakeVLM(),
        confidence_threshold=0.3,
        clean_threshold=0.8,
        partial_threshold=0.4,
        max_segment_span=30.0,
    )

    # Two segments, both with extracted claims, all verdicts "supported".
    assert report.summary.total_descriptions == 2
    assert report.summary.descriptions_flagged == 0
    assert report.metadata.backend == "fake/1"
    assert report.metadata.confidence_threshold == 0.3

    # Second segment had no end → orchestration filled it from segment[0+1]'s
    # start; that's the first segment in our list (5.0 -> uses next which is
    # None -> uses duration, but here we never probed). Actually here seg[1]
    # is the last segment and has no end, so it falls back to duration.
    # Verify end_inferred is recorded for the second segment.
    assert report.segments[0].end_inferred is False
    assert report.segments[1].end_inferred is True


def test_pipeline_does_not_probe_duration_when_no_segment_needs_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If every segment has timestamp_end, ffprobe is never called."""
    descs = tmp_path / "d.json"
    descs.write_text(
        json.dumps(
            [
                {"timestamp_start": 1.0, "timestamp_end": 3.0, "description": "a dog"},
                {"timestamp_start": 4.0, "timestamp_end": 6.0, "description": "a cat"},
            ]
        )
    )
    fake_video = tmp_path / "v.mp4"
    fake_video.write_bytes(b"x")
    monkeypatch.setenv("VIDAUDIT_CACHE_DIR", str(tmp_path / "cache"))

    img = Image.new("RGB", (4, 4), color="red")
    monkeypatch.setattr(
        cli,
        "sample_frames",
        lambda _p, ts, context_window=1.0: {t: [img] for t in ts},
    )

    duration_calls: list[int] = []

    def boom() -> float:
        duration_calls.append(1)
        return 0.0

    monkeypatch.setattr(cli, "get_video_duration", boom)

    run_audit_pipeline(
        video_path=fake_video,
        descriptions_path=descs,
        vlm=_FakeVLM(),
        confidence_threshold=0.3,
        clean_threshold=0.8,
        partial_threshold=0.4,
        max_segment_span=30.0,
    )

    assert duration_calls == []  # never probed


# ---- CLI surface (help / parse) ------------------------------------------


def test_cli_top_level_help_lists_both_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"], env={"COLUMNS": "200"})

    assert result.exit_code == 0
    assert "audit" in result.output
    assert "parse" in result.output


def test_cli_audit_help_lists_thresholds() -> None:
    runner = CliRunner()
    # Widen the terminal so Typer's help formatter doesn't truncate option
    # names into "..." — the assertions below need the full flag strings.
    result = runner.invoke(app, ["audit", "--help"], env={"COLUMNS": "200"})

    assert result.exit_code == 0
    assert "--confidence-threshold" in result.output
    assert "--clean-threshold" in result.output
    assert "--partial-threshold" in result.output
    assert "--max-segment-span" in result.output


def test_cli_parse_prints_claims(tmp_path: Path, descriptions_file: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["parse", "--descriptions", str(descriptions_file)])

    assert result.exit_code == 0
    # Each segment description should appear in the parse output.
    assert "A dog runs" in result.output
    assert "A child laughs" in result.output


def test_cli_audit_help_lists_qwen_flags() -> None:
    """--qwen-revision and --qwen-4bit must surface in audit --help."""
    runner = CliRunner()
    result = runner.invoke(app, ["audit", "--help"], env={"COLUMNS": "200"})

    assert result.exit_code == 0
    assert "--qwen-revision" in result.output
    assert "--qwen-4bit" in result.output


def test_build_backend_qwen_uses_injected_runner_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke test the wiring without loading the real model.

    We monkeypatch ``QwenVLBackend`` itself with a sentinel that records the
    kwargs it receives — proves the CLI forwards --qwen-revision / --qwen-4bit
    correctly without paying the multi-GB transformers import cost.
    """
    seen: dict[str, object] = {}

    class _FakeQwen:
        def __init__(self, **kwargs: object) -> None:
            seen.update(kwargs)
            self.model_id = "fake/qwen"

    monkeypatch.setattr(cli, "QwenVLBackend", _FakeQwen)

    backend = cli._build_backend(
        cli.Backend.qwen,
        model=None,
        min_interval=0.0,
        qwen_revision="abc123",
        qwen_4bit=True,
    )

    assert backend.model_id == "fake/qwen"
    assert seen["revision"] == "abc123"
    assert seen["load_in_4bit"] is True
    assert seen["model"] == "Qwen/Qwen2.5-VL-3B-Instruct"  # default from model_from_env()
