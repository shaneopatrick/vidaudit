"""Tests for the Qwen2.5-VL backend.

The HuggingFace runner is mocked via the ``runner=`` injection seam — no
``transformers``, no ``torch``, no model download. The runner
is a plain callable ``(image, prompt) -> str``, so a tiny stub stands in
for the multi-GB model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from PIL import Image

from vidaudit.vlm.base import VerificationResult
from vidaudit.vlm.qwen_vl import QwenVLBackend, model_from_env

if TYPE_CHECKING:
    from pathlib import Path


class _RecordingRunner:
    """Stub runner that returns canned text and records every prompt it sees."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, image: Image.Image, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("runner called more times than expected")
        return self._responses.pop(0)


@pytest.fixture
def png_image() -> Image.Image:
    return Image.new("RGB", (4, 4), color="blue")


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test gets its own verification cache dir."""
    monkeypatch.setenv("VIDAUDIT_CACHE_DIR", str(tmp_path / "cache"))


def test_verify_claim_returns_verification_result(png_image: Image.Image) -> None:
    runner = _RecordingRunner(
        ['{"verdict": "supported", "confidence": 0.91, "evidence": "a dog is visible"}']
    )
    backend = QwenVLBackend(runner=runner)

    result = backend.verify_claim(png_image, "a dog")

    assert isinstance(result, VerificationResult)
    assert result.claim == "a dog"
    assert result.verdict == "supported"
    assert result.confidence == 0.91
    assert "dog" in result.evidence
    assert len(runner.prompts) == 1
    assert "a dog" in runner.prompts[0]


def test_verify_claim_strips_markdown_fence(png_image: Image.Image) -> None:
    """Qwen sometimes wraps JSON in ```json fences; parser must strip them."""
    runner = _RecordingRunner(
        ['```json\n{"verdict": "unsupported", "confidence": 0.7, "evidence": "no dog"}\n```']
    )
    backend = QwenVLBackend(runner=runner)

    result = backend.verify_claim(png_image, "a dog")

    assert result.verdict == "unsupported"
    assert result.confidence == 0.7


def test_verify_claim_extracts_object_from_surrounding_prose(png_image: Image.Image) -> None:
    """If the model adds a sentence before/after the JSON, we still parse it."""
    runner = _RecordingRunner(
        [
            "Looking at the frame, my answer is:\n"
            '{"verdict": "supported", "confidence": 0.8, "evidence": "visible"}\n'
            "Hope that helps!"
        ]
    )
    backend = QwenVLBackend(runner=runner)

    result = backend.verify_claim(png_image, "a tree")

    assert result.verdict == "supported"
    assert result.confidence == 0.8


def test_verify_claim_cache_hit_skips_runner(png_image: Image.Image) -> None:
    runner = _RecordingRunner(['{"verdict": "supported", "confidence": 0.9, "evidence": "warm"}'])
    backend = QwenVLBackend(runner=runner)

    first = backend.verify_claim(png_image, "a cat")
    second = backend.verify_claim(png_image, "a cat")

    assert len(runner.prompts) == 1  # second call served from cache
    assert second == first


def test_verify_claim_malformed_json_falls_back_to_uncertain(png_image: Image.Image) -> None:
    runner = _RecordingRunner(["this is definitely not JSON at all"])
    backend = QwenVLBackend(runner=runner)

    result = backend.verify_claim(png_image, "something")

    assert result.verdict == "uncertain"
    assert result.confidence == 0.0
    assert "parseable" in result.evidence.lower() or "json" in result.evidence.lower()


def test_verify_batch_returns_in_input_order(png_image: Image.Image) -> None:
    response = (
        "["
        '{"claim": "red jacket", "verdict": "supported", "confidence": 0.9, "evidence": "x"},'
        '{"claim": "coffee cup", "verdict": "unsupported", "confidence": 0.6, "evidence": "y"}'
        "]"
    )
    runner = _RecordingRunner([response])
    backend = QwenVLBackend(runner=runner)

    results = backend.verify_batch(png_image, ["red jacket", "coffee cup"])

    assert [r.claim for r in results] == ["red jacket", "coffee cup"]
    assert [r.verdict for r in results] == ["supported", "unsupported"]
    # Batch prompt is one call, not one-per-claim.
    assert len(runner.prompts) == 1
    assert "red jacket" in runner.prompts[0]
    assert "coffee cup" in runner.prompts[0]


def test_verify_batch_aligns_positionally_when_claim_paraphrased(
    png_image: Image.Image,
) -> None:
    """Qwen sometimes paraphrases the echoed claim ("the dog" → "dog")."""
    runner = _RecordingRunner(
        [
            "[\n"
            '  {"claim": "dog", "verdict": "supported", "confidence": 0.9, "evidence": "x"},\n'
            '  {"claim": "ball", "verdict": "supported", "confidence": 0.8, "evidence": "y"}\n'
            "]"
        ]
    )
    backend = QwenVLBackend(runner=runner)

    results = backend.verify_batch(png_image, ["the dog", "a ball"])

    # Both claims should resolve via positional fallback (lengths matched).
    assert [r.verdict for r in results] == ["supported", "supported"]
    assert results[0].claim == "the dog"  # claim text preserved on VerificationResult


def test_verify_batch_handles_omitted_claim_as_uncertain(png_image: Image.Image) -> None:
    response = (
        '[{"claim": "red jacket", "verdict": "supported", "confidence": 0.9, "evidence": "x"}]'
    )
    runner = _RecordingRunner([response])
    backend = QwenVLBackend(runner=runner)

    results = backend.verify_batch(png_image, ["red jacket", "coffee cup"])

    assert results[1].claim == "coffee cup"
    assert results[1].verdict == "uncertain"
    assert results[1].confidence == 0.0


def test_verify_batch_only_calls_runner_for_uncached_claims(png_image: Image.Image) -> None:
    runner = _RecordingRunner(
        [
            # Warm cache for "red jacket"
            '{"verdict": "supported", "confidence": 0.9, "evidence": "warm"}',
            # Batch call should ask only for "coffee cup"
            "["
            '{"claim": "coffee cup", "verdict": "unsupported", "confidence": 0.7, "evidence": "y"}'
            "]",
        ]
    )
    backend = QwenVLBackend(runner=runner)

    backend.verify_claim(png_image, "red jacket")
    results = backend.verify_batch(png_image, ["red jacket", "coffee cup"])

    assert len(runner.prompts) == 2  # one warm + one batch
    assert "coffee cup" in runner.prompts[1]
    assert "red jacket" not in runner.prompts[1]  # cached, not re-sent
    assert results[0].evidence == "warm"  # served from cache
    assert results[1].verdict == "unsupported"


def test_model_id_includes_revision_when_pinned() -> None:
    runner = _RecordingRunner([])
    backend = QwenVLBackend(runner=runner, revision="abc123def")

    assert backend.model_id == "Qwen/Qwen2.5-VL-3B-Instruct@abc123def"


def test_model_id_omits_revision_when_unpinned() -> None:
    runner = _RecordingRunner([])
    backend = QwenVLBackend(runner=runner)

    assert backend.model_id == "Qwen/Qwen2.5-VL-3B-Instruct"


def test_model_from_env_respects_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIDAUDIT_QWEN_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    assert model_from_env() == "Qwen/Qwen2.5-VL-7B-Instruct"


def test_model_from_env_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIDAUDIT_QWEN_MODEL", raising=False)
    assert model_from_env() == "Qwen/Qwen2.5-VL-3B-Instruct"
