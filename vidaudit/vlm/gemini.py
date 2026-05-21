"""Gemini 2.5 Flash backend for claim verification.

Closed-source dev/no-GPU fallback (DESIGN.md DD-16). The canonical eval runs
on the Qwen open-weight backend; this backend exists for local iteration on
machines without GPUs and as a comparison point in the cross-backend eval
(DD-13).

Uses Gemini's native structured output: a Pydantic :class:`_GeminiVerdict`
(or ``list[_GeminiBatchVerdict]``) is passed as ``response_schema`` so the
SDK returns validated, typed objects via ``response.parsed`` (DD-10). No
prompt-engineered JSON format hints, no regex fallback as the primary path.
"""

from __future__ import annotations

import logging
import os
import time
from io import BytesIO
from typing import TYPE_CHECKING, Literal

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field

from vidaudit.vlm._cache import cache_get, cache_key, cache_put
from vidaudit.vlm.base import VerificationResult, VLMBackend

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"


_CONFIDENCE_DESCRIPTION = (
    "Your confidence in the verdict (1.0 = certain, 0.0 = pure guess). This "
    "is NOT the probability that the claim is true — it is your confidence "
    "in the verdict label you just gave."
)
_EVIDENCE_DESCRIPTION = (
    "One short sentence describing what you see or do not see in the frame "
    "that supports this verdict."
)


class _GeminiVerdict(BaseModel):
    """Response schema for a single-claim Gemini call (DD-10)."""

    verdict: Literal["supported", "unsupported", "uncertain"] = Field(
        description="Whether the claim is visually supported by the frame."
    )
    confidence: float = Field(ge=0.0, le=1.0, description=_CONFIDENCE_DESCRIPTION)
    evidence: str = Field(description=_EVIDENCE_DESCRIPTION)


class _GeminiBatchVerdict(BaseModel):
    """One item in a batched Gemini response.

    Echoes the claim text so we can correlate verdicts to inputs robustly
    even if the model normalises or reorders them.
    """

    claim: str = Field(
        description="The claim text being verified, echoed verbatim from the input."
    )
    verdict: Literal["supported", "unsupported", "uncertain"] = Field(
        description="Whether the claim is visually supported by the frame."
    )
    confidence: float = Field(ge=0.0, le=1.0, description=_CONFIDENCE_DESCRIPTION)
    evidence: str = Field(description=_EVIDENCE_DESCRIPTION)


_SINGLE_PROMPT = (
    "You are a video frame auditor. You are shown one frame from a video and "
    "one claim about what appears in that frame. Decide whether the claim is "
    "visually supported by the frame.\n\n"
    'Claim: "{claim}"'
)


def _batch_prompt(claims: list[str]) -> str:
    numbered = "\n".join(f'{i + 1}. "{c}"' for i, c in enumerate(claims))
    return (
        "You are a video frame auditor. You are shown one frame from a video "
        "and a list of claims about what appears in that frame. For EACH "
        "claim, decide whether it is visually supported by the frame. Echo "
        "each claim back exactly as given; return one verdict per claim, in "
        "the same order as the input.\n\n"
        "Claims:\n" + numbered
    )


class GeminiBackend(VLMBackend):
    """Gemini 2.5 Flash backend.

    Args:
        model: Gemini model identifier. Defaults to ``gemini-2.5-flash``.
        api_key: Override the ``GEMINI_API_KEY`` environment variable.
        min_interval_seconds: Minimum wall-clock interval between API calls
            (free-tier rate-limit pacing). Default 4.0s (≈15 RPM).
        max_retries: Retries on transient API errors (429 / 5xx). Default 3.
        client: Optional injected client (for tests).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        min_interval_seconds: float = 4.0,
        max_retries: int = 3,
        client: genai.Client | None = None,
    ) -> None:
        if client is None:
            key = api_key or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("Set GEMINI_API_KEY environment variable (or pass api_key=).")
            client = genai.Client(api_key=key)
        self._client = client
        self.model_id = model
        self._min_interval = min_interval_seconds
        self._max_retries = max_retries
        self._last_call: float = 0.0

    def verify_claim(self, image: Image.Image, claim: str) -> VerificationResult:
        key = cache_key(image, claim, self.model_id)
        cached = cache_get(key)
        if cached is not None:
            return cached

        parsed = self._call_with_retry(
            schema=_GeminiVerdict,
            prompt=_SINGLE_PROMPT.format(claim=claim),
            image=image,
        )
        if isinstance(parsed, _GeminiVerdict):
            result = VerificationResult(claim=claim, **parsed.model_dump())
        else:
            result = _uncertain(claim, "VLM returned no parseable response.")
        cache_put(key, result)
        return result

    def verify_batch(self, image: Image.Image, claims: list[str]) -> list[VerificationResult]:
        by_claim: dict[str, VerificationResult] = {}
        uncached: list[str] = []
        for c in claims:
            cached = cache_get(cache_key(image, c, self.model_id))
            if cached is not None:
                by_claim[c] = cached
            else:
                uncached.append(c)

        if uncached:
            parsed = self._call_with_retry(
                schema=list[_GeminiBatchVerdict],
                prompt=_batch_prompt(uncached),
                image=image,
            )
            by_text = _index_batch_response(parsed, uncached)
            for c in uncached:
                item = by_text.get(c)
                if item is None:
                    result = _uncertain(c, "VLM omitted this claim from the batched response.")
                else:
                    result = VerificationResult(**item.model_dump())
                cache_put(cache_key(image, c, self.model_id), result)
                by_claim[c] = result

        return [by_claim[c] for c in claims]

    def _call_with_retry(
        self,
        schema: object,
        prompt: str,
        image: Image.Image,
    ) -> object:
        """Send (prompt, image) with structured output. Returns ``response.parsed``.

        Retries on 429 / 5xx with exponential backoff; non-retriable errors
        propagate. Returns ``None`` if the SDK ultimately yields no parseable
        structured output (caller treats this as ``uncertain``).
        """
        contents = [
            types.Part.from_bytes(data=_to_png_bytes(image), mime_type="image/png"),
            types.Part.from_text(text=prompt),
        ]
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.0,
        )

        for attempt in range(self._max_retries):
            self._wait_for_rate_limit()
            try:
                response = self._client.models.generate_content(
                    model=self.model_id,
                    contents=contents,  # type: ignore[arg-type]  # list[Part] is fine; SDK union is invariant
                    config=config,
                )
            except genai_errors.APIError as exc:
                code = getattr(exc, "code", None)
                retriable = code == 429 or (isinstance(code, int) and 500 <= code < 600)
                if not retriable or attempt == self._max_retries - 1:
                    raise
                backoff = 2**attempt
                logger.warning(
                    "Gemini APIError (code=%s) attempt %d/%d — backoff %ds: %s",
                    code,
                    attempt + 1,
                    self._max_retries,
                    backoff,
                    exc,
                )
                time.sleep(backoff)
                continue

            parsed = response.parsed
            if parsed is not None:
                return parsed
            logger.warning(
                "Gemini returned no parseable structured output (attempt %d/%d)",
                attempt + 1,
                self._max_retries,
            )
        return None

    def _wait_for_rate_limit(self) -> None:
        if self._min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()


def _uncertain(claim: str, reason: str) -> VerificationResult:
    return VerificationResult(claim=claim, verdict="uncertain", confidence=0.0, evidence=reason)


def _index_batch_response(parsed: object, requested: list[str]) -> dict[str, _GeminiBatchVerdict]:
    """Build a {claim_text: item} index, with positional fallback if lengths match."""
    if not isinstance(parsed, list):
        return {}
    items = [x for x in parsed if isinstance(x, _GeminiBatchVerdict)]
    by_text: dict[str, _GeminiBatchVerdict] = {item.claim: item for item in items}
    if len(items) == len(requested):
        for c, item in zip(requested, items, strict=False):
            by_text.setdefault(c, item)
    return by_text


def _to_png_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
