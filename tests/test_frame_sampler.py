"""Tests for vidaudit.frame_sampler.

ffmpeg/ffprobe subprocess calls are mocked — tests must
never require a real video file. A dummy video file is created so existence
checks pass; everything downstream is faked.
"""

from __future__ import annotations

import logging
import subprocess
from io import BytesIO
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from vidaudit import frame_sampler
from vidaudit.frame_sampler import sample_frames

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    FakeRun = Callable[..., subprocess.CompletedProcess[object]]


def _png_bytes() -> bytes:
    """Tiny valid PNG, used as the canned ffmpeg stdout in tests."""
    img = Image.new("RGB", (4, 4), color="red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fake_run_factory(
    duration: float = 10.0,
    ffprobe_returncode: int = 0,
    ffmpeg_returncode: int = 0,
) -> tuple[FakeRun, list[list[str]]]:
    """Return (fake subprocess.run, captured-calls list)."""
    png = _png_bytes()
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[object]:
        calls.append(list(cmd))
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(
                cmd,
                ffprobe_returncode,
                stdout=(f"{duration}\n" if ffprobe_returncode == 0 else ""),
                stderr=("" if ffprobe_returncode == 0 else "probe error"),
            )
        return subprocess.CompletedProcess(
            cmd,
            ffmpeg_returncode,
            stdout=(png if ffmpeg_returncode == 0 else b""),
            stderr=(b"" if ffmpeg_returncode == 0 else b"ffmpeg error"),
        )

    return fake_run, calls


@pytest.fixture
def video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a dummy video file and point the cache at tmp_path."""
    monkeypatch.setenv("VIDAUDIT_CACHE_DIR", str(tmp_path / "cache"))
    path = tmp_path / "sample.mp4"
    path.write_bytes(b"not really a video")
    return path


def test_sample_frames_returns_primary_plus_context(
    video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_run, _ = _fake_run_factory(duration=10.0)
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = sample_frames(video, [5.0], context_window=1.0)

    assert list(result.keys()) == [5.0]
    assert len(result[5.0]) == 3
    assert all(isinstance(img, Image.Image) for img in result[5.0])


def test_sample_frames_uses_ss_after_i_for_accurate_seek(
    video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`-ss` must come AFTER `-i`, or ffmpeg snaps to the nearest keyframe."""
    fake_run, calls = _fake_run_factory()
    monkeypatch.setattr(subprocess, "run", fake_run)

    sample_frames(video, [5.0], context_window=1.0)

    ffmpeg_calls = [c for c in calls if c[0] == "ffmpeg"]
    assert ffmpeg_calls, "expected at least one ffmpeg invocation"
    for cmd in ffmpeg_calls:
        assert cmd.index("-i") < cmd.index("-ss")


def test_sample_frames_subprocess_called_with_list_form(
    video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never shell-string a subprocess command — list form only."""
    captured: list[dict[str, object]] = []

    def recording_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[object]:
        captured.append({"cmd": cmd, "shell": kwargs.get("shell", False)})
        fake_run, _ = _fake_run_factory()
        return fake_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording_run)

    sample_frames(video, [5.0])

    for call in captured:
        assert isinstance(call["cmd"], list)
        assert call["shell"] is False


def test_sample_frames_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sample_frames(tmp_path / "nope.mp4", [1.0])


def test_sample_frames_skips_timestamp_beyond_duration(
    video: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_run, _ = _fake_run_factory(duration=10.0)
    monkeypatch.setattr(subprocess, "run", fake_run)

    with caplog.at_level(logging.WARNING):
        result = sample_frames(video, [5.0, 15.0])

    assert set(result.keys()) == {5.0}
    assert "outside video duration" in caplog.text


def test_sample_frames_clamps_negative_offsets(
    video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_run, calls = _fake_run_factory(duration=10.0)
    monkeypatch.setattr(subprocess, "run", fake_run)

    sample_frames(video, [0.5], context_window=1.0)

    ffmpeg_ss_values = [float(cmd[cmd.index("-ss") + 1]) for cmd in calls if cmd[0] == "ffmpeg"]
    # Offsets [0.5, -0.5 -> 0.0, 1.5] should all be present; nothing negative.
    assert ffmpeg_ss_values == [0.5, 0.0, 1.5]
    assert all(v >= 0.0 for v in ffmpeg_ss_values)


def test_sample_frames_dedupes_clamped_collisions(
    video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At t=0.0 the t-1s context clamps onto the primary; the duplicate is dropped."""
    fake_run, calls = _fake_run_factory(duration=10.0)
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = sample_frames(video, [0.0], context_window=1.0)

    assert len(result[0.0]) == 2  # [0.0, 1.0], with -1.0 clamped to 0.0 and dropped
    ffmpeg_calls = [c for c in calls if c[0] == "ffmpeg"]
    assert len(ffmpeg_calls) == 2


def test_sample_frames_caches_decoded_frames(video: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_run, calls = _fake_run_factory(duration=10.0)
    monkeypatch.setattr(subprocess, "run", fake_run)

    sample_frames(video, [5.0], context_window=1.0)
    first_ffmpeg = sum(1 for c in calls if c[0] == "ffmpeg")
    calls.clear()

    sample_frames(video, [5.0], context_window=1.0)
    second_ffmpeg = sum(1 for c in calls if c[0] == "ffmpeg")

    assert first_ffmpeg == 3
    assert second_ffmpeg == 0  # all three frames served from cache


def test_sample_frames_ffprobe_failure_raises_runtime_error(
    video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_run, _ = _fake_run_factory(ffprobe_returncode=1)
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="ffprobe failed"):
        sample_frames(video, [5.0])


def test_sample_frames_ffmpeg_failure_raises_runtime_error(
    video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_run, _ = _fake_run_factory(ffmpeg_returncode=1)
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        sample_frames(video, [5.0])


def test_sample_frames_skips_failing_context_frame_but_keeps_primary(
    video: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A context frame that fails extraction is skipped; the primary survives."""
    png = _png_bytes()

    def fake_run(
        cmd: list[str], *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[object]:
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, stdout="10.0\n", stderr="")
        ss = float(cmd[cmd.index("-ss") + 1])
        if ss > 9.0:  # the upper context frame near EOF fails
            return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"eof")
        return subprocess.CompletedProcess(cmd, 0, stdout=png, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    # t=8.5, window=1.0 -> [8.5, 7.5, 9.5]; the 9.5 context frame fails.
    with caplog.at_level(logging.WARNING):
        result = sample_frames(video, [8.5], context_window=1.0)

    assert len(result[8.5]) == 2  # primary + one good context; failing one dropped
    assert "Skipping context frame" in caplog.text


def test_sample_frames_raises_when_primary_frame_fails(
    video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Primary-frame failure is fatal even though context frames are best-effort."""
    png = _png_bytes()

    def fake_run(
        cmd: list[str], *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[object]:
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, stdout="10.0\n", stderr="")
        ss = float(cmd[cmd.index("-ss") + 1])
        if abs(ss - 5.0) < 1e-6:  # the primary frame fails
            return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"bad")
        return subprocess.CompletedProcess(cmd, 0, stdout=png, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        sample_frames(video, [5.0], context_window=1.0)


def test_module_imports_clean() -> None:
    # Sanity check the module loaded without side effects.
    assert hasattr(frame_sampler, "sample_frames")
