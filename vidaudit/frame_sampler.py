"""Extract frames from a video at specific timestamps via ffmpeg subprocess.

Frame fidelity is the whole game for an auditor — sampling the wrong frame
manufactures false hallucinations — so seeking is frame-accurate (``-ss`` after
``-i``, DESIGN.md DD-8) and decoded frames are cached by
(video identity, timestamp) under ``VIDAUDIT_CACHE_DIR`` so reruns are
deterministic and don't re-decode (DD-14).
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from io import BytesIO
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path(".vidaudit_cache")


def _cache_dir() -> Path:
    """Resolve and ensure the frames cache directory.

    Honors ``VIDAUDIT_CACHE_DIR`` if set (CLAUDE.md §7), defaulting to
    ``.vidaudit_cache/`` next to the working directory.
    """
    env = os.environ.get("VIDAUDIT_CACHE_DIR")
    base = Path(env) if env else _DEFAULT_CACHE_DIR
    frames = base / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    return frames


def _cache_key(video_path: Path, t: float) -> str:
    """Stable per-frame cache key tied to video identity and timestamp.

    Includes size and mtime so a video replaced at the same path invalidates
    the cache automatically.
    """
    stat = video_path.stat()
    raw = f"{video_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{t:.6f}"
    return hashlib.sha1(raw.encode()).hexdigest()


def get_video_duration(video_path: Path) -> float:
    """Return the video duration in seconds via ffprobe.

    Raises:
        RuntimeError: If ffprobe fails or returns unparseable output
            (corrupted or unreadable video).
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            f"Could not read video (ffprobe failed): {video_path}\n{result.stderr.strip()}"
        )
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"Could not parse ffprobe duration for {video_path}: {result.stdout!r}"
        ) from exc


def _extract_one(video_path: Path, t: float) -> Image.Image:
    """Extract a single frame at exactly ``t`` seconds, with caching.

    ``-ss`` is placed AFTER ``-i`` for frame-accurate seeking (DD-8); the
    fast-seek form would snap to the nearest keyframe.
    """
    cache_path = _cache_dir() / f"{_cache_key(video_path, t)}.png"
    if cache_path.exists():
        with Image.open(cache_path) as cached:
            return cached.copy()

    result = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-ss",
            f"{t:.6f}",
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(
            f"ffmpeg failed to extract frame at t={t:.3f}s from {video_path}: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )

    with Image.open(BytesIO(result.stdout)) as raw:
        image = raw.copy()
    image.save(cache_path, format="PNG")
    return image


def sample_frames(
    video_path: Path,
    timestamps: list[float],
    context_window: float = 1.0,
) -> dict[float, list[Image.Image]]:
    """Extract frames at each timestamp, plus neighbouring context frames.

    For each timestamp ``t`` returns three frames: the primary at ``t`` and
    context at ``t ± context_window``. Offsets outside ``[0, duration]`` are
    clamped, then collapsed if clamping produces duplicates. Timestamps whose
    primary frame falls outside the video's duration are skipped with a
    warning (PLAN edge case). Frames are cached under ``VIDAUDIT_CACHE_DIR``.

    The primary frame is always at index 0 of each returned list — callers
    rely on this contract (see ``auditors/object_audit.py``).

    Args:
        video_path: Path to the video file (mp4, mov, mkv, …).
        timestamps: Timestamps in seconds.
        context_window: Seconds before/after to sample as context. Default 1.0.

    Returns:
        Mapping from each kept timestamp to its ``[primary, *context]`` frames.

    Raises:
        FileNotFoundError: If ``video_path`` does not exist.
        RuntimeError: If the video cannot be read or ffmpeg fails on a frame.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    duration = get_video_duration(video_path)

    frames: dict[float, list[Image.Image]] = {}
    for t in timestamps:
        if t < 0 or t > duration:
            logger.warning(
                "Skipping timestamp %.3fs outside video duration [0, %.3f]",
                t,
                duration,
            )
            continue

        # Primary first so callers can rely on index 0 = frame at t.
        offsets = [t, t - context_window, t + context_window]
        seen: set[float] = set()
        clip_times: list[float] = []
        for o in offsets:
            clipped = max(0.0, min(o, duration))
            key = round(clipped, 6)
            if key in seen:
                continue
            seen.add(key)
            clip_times.append(clipped)

        frames[t] = [_extract_one(video_path, ct) for ct in clip_times]

    return frames
