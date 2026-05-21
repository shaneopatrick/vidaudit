"""Tests for the VLMBackend ABC and VerificationResult model."""

from __future__ import annotations

import pytest
from PIL import Image
from pydantic import ValidationError

from vidaudit.vlm.base import VerificationResult, VLMBackend


class _DummyBackend(VLMBackend):
    """Records calls so we can verify default verify_batch behaviour."""

    def __init__(self) -> None:
        self.model_id = "dummy/0"
        self.calls: list[str] = []

    def verify_claim(self, image: Image.Image, claim: str) -> VerificationResult:
        self.calls.append(claim)
        return VerificationResult(
            claim=claim, verdict="supported", confidence=1.0, evidence="dummy"
        )


def test_verification_result_round_trip() -> None:
    result = VerificationResult(
        claim="a woman", verdict="supported", confidence=0.9, evidence="visible"
    )
    restored = VerificationResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_verification_result_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        VerificationResult(claim="x", verdict="supported", confidence=1.5)
    with pytest.raises(ValidationError):
        VerificationResult(claim="x", verdict="supported", confidence=-0.1)


def test_verification_result_evidence_defaults_to_empty() -> None:
    result = VerificationResult(claim="x", verdict="uncertain", confidence=0.0)
    assert result.evidence == ""


def test_default_verify_batch_loops_over_verify_claim() -> None:
    backend = _DummyBackend()
    image = Image.new("RGB", (4, 4), "red")

    results = backend.verify_batch(image, ["a", "b", "c"])

    assert backend.calls == ["a", "b", "c"]
    assert [r.claim for r in results] == ["a", "b", "c"]
    assert all(r.verdict == "supported" for r in results)
