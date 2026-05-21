"""Verification-result cache shared across VLM backends (DESIGN.md DD-11).

Keyed by SHA-1 of (frame PNG bytes, claim text, model identifier). Cached
results are serialized :class:`VerificationResult` JSON so they round-trip
cleanly. The model identifier in the key means a backend swap (Gemini → Qwen)
does not invalidate cached verdicts from the other backend.
"""

from __future__ import annotations

import hashlib
import os
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from vidaudit.vlm.base import VerificationResult

if TYPE_CHECKING:
    from PIL import Image


def cache_dir() -> Path:
    """Resolve and ensure the verification cache directory.

    Honors ``VIDAUDIT_CACHE_DIR`` (CLAUDE.md §7), defaulting to
    ``.vidaudit_cache/`` next to the working directory.
    """
    env = os.environ.get("VIDAUDIT_CACHE_DIR")
    base = Path(env) if env else Path(".vidaudit_cache")
    sub = base / "verifications"
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def cache_key(image: Image.Image, claim: str, model_id: str) -> str:
    """Stable per-(frame, claim, model) cache key.

    SHA-1 over PNG-encoded frame bytes plus the claim text plus the model
    identifier. PNG encoding is deterministic enough for caching given the
    upstream frame_sampler already round-trips frames through PNG.
    """
    buf = BytesIO()
    image.save(buf, format="PNG")
    h = hashlib.sha1()
    h.update(buf.getvalue())
    h.update(b"|")
    h.update(claim.encode("utf-8"))
    h.update(b"|")
    h.update(model_id.encode("utf-8"))
    return h.hexdigest()


def cache_get(key: str) -> VerificationResult | None:
    """Return the cached result for ``key``, or ``None`` if not cached."""
    path = cache_dir() / f"{key}.json"
    if not path.exists():
        return None
    return VerificationResult.model_validate_json(path.read_bytes())


def cache_put(key: str, result: VerificationResult) -> None:
    """Persist ``result`` under ``key``."""
    path = cache_dir() / f"{key}.json"
    path.write_bytes(result.model_dump_json().encode("utf-8"))
