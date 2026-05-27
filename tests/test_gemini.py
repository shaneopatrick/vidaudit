"""Tests for the Gemini backend.

The google-genai SDK is mocked — tests must never hit the real API. The
backend takes an optional ``client=`` parameter for exactly this purpose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from PIL import Image

from vidaudit.vlm.base import VerificationResult
from vidaudit.vlm.gemini import (
    GeminiBackend,
    _GeminiBatchVerdict,
    _GeminiVerdict,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def png_image() -> Image.Image:
    return Image.new("RGB", (4, 4), color="red")


@pytest.fixture
def gemini(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[GeminiBackend, MagicMock]:
    monkeypatch.setenv("VIDAUDIT_CACHE_DIR", str(tmp_path / "cache"))
    mock_client = MagicMock()
    backend = GeminiBackend(client=mock_client, min_interval_seconds=0.0, max_retries=2)
    return backend, mock_client


def _single_response(
    verdict: str = "supported", confidence: float = 0.9, evidence: str = "ok"
) -> MagicMock:
    response = MagicMock()
    response.parsed = _GeminiVerdict(
        verdict=verdict,  # type: ignore[arg-type]
        confidence=confidence,
        evidence=evidence,
    )
    return response


def _batch_response(items: list[_GeminiBatchVerdict]) -> MagicMock:
    response = MagicMock()
    response.parsed = items
    return response


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiBackend()


def test_verify_claim_returns_verification_result(
    gemini: tuple[GeminiBackend, MagicMock], png_image: Image.Image
) -> None:
    backend, client = gemini
    client.models.generate_content.return_value = _single_response(
        verdict="supported", confidence=0.95, evidence="A woman is visible."
    )

    result = backend.verify_claim(png_image, "a woman")

    assert isinstance(result, VerificationResult)
    assert result.claim == "a woman"
    assert result.verdict == "supported"
    assert result.confidence == 0.95
    assert result.evidence == "A woman is visible."
    client.models.generate_content.assert_called_once()


def test_verify_claim_passes_pydantic_model_as_response_schema(
    gemini: tuple[GeminiBackend, MagicMock], png_image: Image.Image
) -> None:
    """The schema must be the Pydantic class itself, not a JSON-Schema dict."""
    backend, client = gemini
    client.models.generate_content.return_value = _single_response()

    backend.verify_claim(png_image, "anything")

    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.response_schema is _GeminiVerdict
    assert config.response_mime_type == "application/json"
    assert config.temperature == 0.0


def test_verify_claim_cache_hit_skips_sdk(
    gemini: tuple[GeminiBackend, MagicMock], png_image: Image.Image
) -> None:
    backend, client = gemini
    client.models.generate_content.return_value = _single_response()

    backend.verify_claim(png_image, "a woman")
    client.models.generate_content.reset_mock()
    second = backend.verify_claim(png_image, "a woman")

    client.models.generate_content.assert_not_called()
    assert second.claim == "a woman"


def test_verify_claim_none_parsed_falls_back_to_uncertain(
    gemini: tuple[GeminiBackend, MagicMock], png_image: Image.Image
) -> None:
    backend, client = gemini
    response = MagicMock()
    response.parsed = None
    client.models.generate_content.return_value = response

    result = backend.verify_claim(png_image, "anything")

    assert result.verdict == "uncertain"
    assert result.confidence == 0.0
    # max_retries=2 in the fixture, so two attempts before giving up.
    assert client.models.generate_content.call_count == 2


def test_verify_batch_uses_batch_schema_and_returns_in_input_order(
    gemini: tuple[GeminiBackend, MagicMock], png_image: Image.Image
) -> None:
    backend, client = gemini
    client.models.generate_content.return_value = _batch_response(
        [
            _GeminiBatchVerdict(
                claim="red jacket",
                verdict="supported",
                confidence=0.9,
                evidence="visible",
            ),
            _GeminiBatchVerdict(
                claim="coffee cup",
                verdict="unsupported",
                confidence=0.7,
                evidence="absent",
            ),
        ]
    )

    results = backend.verify_batch(png_image, ["red jacket", "coffee cup"])

    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.response_schema == list[_GeminiBatchVerdict]
    assert [r.claim for r in results] == ["red jacket", "coffee cup"]
    assert [r.verdict for r in results] == ["supported", "unsupported"]


def test_verify_batch_serves_cached_claims_and_only_calls_sdk_for_uncached(
    gemini: tuple[GeminiBackend, MagicMock], png_image: Image.Image
) -> None:
    backend, client = gemini
    # Warm the cache for "red jacket" via verify_claim.
    client.models.generate_content.return_value = _single_response(
        verdict="supported", confidence=0.9, evidence="warm"
    )
    backend.verify_claim(png_image, "red jacket")
    client.models.generate_content.reset_mock()

    # Now batch with one cached, one fresh — only "coffee cup" should be sent.
    client.models.generate_content.return_value = _batch_response(
        [
            _GeminiBatchVerdict(
                claim="coffee cup",
                verdict="unsupported",
                confidence=0.6,
                evidence="not visible",
            )
        ]
    )

    results = backend.verify_batch(png_image, ["red jacket", "coffee cup"])

    assert client.models.generate_content.call_count == 1
    prompt = client.models.generate_content.call_args.kwargs["contents"][1].text
    assert "coffee cup" in prompt
    assert "red jacket" not in prompt
    assert [r.claim for r in results] == ["red jacket", "coffee cup"]
    assert results[0].evidence == "warm"  # served from cache


def test_verify_batch_handles_omitted_claim_as_uncertain(
    gemini: tuple[GeminiBackend, MagicMock], png_image: Image.Image
) -> None:
    backend, client = gemini
    # SDK returns only one of two claims — the missing one becomes uncertain.
    client.models.generate_content.return_value = _batch_response(
        [
            _GeminiBatchVerdict(
                claim="red jacket",
                verdict="supported",
                confidence=0.9,
                evidence="visible",
            )
        ]
    )

    results = backend.verify_batch(png_image, ["red jacket", "coffee cup"])

    assert results[1].claim == "coffee cup"
    assert results[1].verdict == "uncertain"
    assert results[1].confidence == 0.0


def test_model_id_is_set(
    gemini: tuple[GeminiBackend, MagicMock],
) -> None:
    backend, _ = gemini
    assert backend.model_id == "gemini-2.5-flash"
