"""Tests for vidaudit.auditors.object_audit."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from PIL import Image

from vidaudit.auditors.object_audit import (
    SegmentAuditResult,
    audit_segment,
)
from vidaudit.description_parser import Claim, DescriptionSegment
from vidaudit.vlm.base import VerificationResult, VLMBackend

# ---- helpers --------------------------------------------------------------

Responder = Callable[[Image.Image, str], VerificationResult]


class _FakeVLM(VLMBackend):
    """Backend driven by a (image, claim) -> VerificationResult callable.

    Overrides ``verify_batch`` directly (rather than inheriting the ABC's
    loop) so tests can distinguish batched calls from per-claim calls —
    required to assert DD-6 batching of context-frame checks. The ABC's
    default loop is covered separately in ``test_vlm_base``.
    """

    def __init__(self, responder: Responder) -> None:
        self.model_id = "fake"
        self._responder = responder
        self.calls: list[tuple[Image.Image, str]] = []
        self.batch_calls: list[tuple[Image.Image, list[str]]] = []
        self.single_calls: list[tuple[Image.Image, str]] = []

    def verify_claim(self, image: Image.Image, claim: str) -> VerificationResult:
        self.calls.append((image, claim))
        self.single_calls.append((image, claim))
        return self._responder(image, claim)

    def verify_batch(self, image: Image.Image, claims: list[str]) -> list[VerificationResult]:
        self.batch_calls.append((image, list(claims)))
        results: list[VerificationResult] = []
        for c in claims:
            self.calls.append((image, c))
            results.append(self._responder(image, c))
        return results


def _image(color: str = "red") -> Image.Image:
    return Image.new("RGB", (4, 4), color=color)


def _claim(text: str) -> Claim:
    return Claim(text=text, claim_type="object", source_description="ignored")


def _segment(*texts: str) -> DescriptionSegment:
    return DescriptionSegment(
        timestamp_start=0.0,
        timestamp_end=5.0,
        description=" ".join(texts) or "ø",
        claims=[_claim(t) for t in texts],
    )


def _supported(claim: str, conf: float = 0.9) -> VerificationResult:
    return VerificationResult(
        claim=claim, verdict="supported", confidence=conf, evidence="visible"
    )


def _unsupported(claim: str, conf: float = 0.9) -> VerificationResult:
    return VerificationResult(
        claim=claim, verdict="unsupported", confidence=conf, evidence="absent"
    )


# ---- tests ----------------------------------------------------------------


def test_all_supported_is_clean_grounding_one() -> None:
    seg = _segment("woman", "coffee cup")
    vlm = _FakeVLM(lambda _img, claim: _supported(claim))

    result = audit_segment(seg, [_image()], vlm)

    assert result.verdict == "clean"
    assert result.grounding_score == 1.0
    assert result.hallucination_count == 0
    assert all(not cr.flagged for cr in result.claim_results)


def test_low_confidence_unsupported_is_neither_flagged_nor_escalated() -> None:
    """DD-7: a low-confidence "unsupported" is uncertain, not a hallucination."""
    seg = _segment("woman")
    primary, ctx = _image("red"), _image("blue")
    vlm = _FakeVLM(lambda _img, claim: _unsupported(claim, conf=0.2))

    result = audit_segment(seg, [primary, ctx], vlm, confidence_threshold=0.3)

    assert result.claim_results[0].flagged is False
    assert result.hallucination_count == 0
    # Only the primary should have been queried — context check is skipped
    # because the primary verdict's confidence is below the threshold.
    assert len(vlm.calls) == 1


def test_confident_unsupported_rescued_by_context() -> None:
    seg = _segment("woman")
    primary, ctx = _image("red"), _image("blue")

    def responder(img: Image.Image, claim: str) -> VerificationResult:
        return _unsupported(claim) if img is primary else _supported(claim)

    vlm = _FakeVLM(responder)
    result = audit_segment(seg, [primary, ctx], vlm)

    assert result.claim_results[0].flagged is False
    assert result.claim_results[0].verification.verdict == "supported"
    assert result.grounding_score == 1.0


def test_confident_unsupported_across_all_frames_is_flagged() -> None:
    seg = _segment("unicorn")
    frames = [_image("red"), _image("blue"), _image("green")]
    vlm = _FakeVLM(lambda _img, claim: _unsupported(claim))

    result = audit_segment(seg, frames, vlm)

    assert result.claim_results[0].flagged is True
    assert result.hallucination_count == 1
    assert result.grounding_score == 0.0
    assert result.verdict == "full_hallucination"


def test_unsupported_with_no_context_frames_is_flagged() -> None:
    seg = _segment("unicorn")
    vlm = _FakeVLM(lambda _img, claim: _unsupported(claim))

    result = audit_segment(seg, [_image()], vlm)

    assert result.claim_results[0].flagged is True
    assert result.hallucination_count == 1


def test_context_checks_are_batched_per_frame() -> None:
    """DD-6: each context frame is ONE batched call, not one call per claim.

    Without batching the audit would issue (#unsupported_claims × #context_frames)
    individual API calls. With batching, every pending claim is sent to each
    context frame in a single ``verify_batch`` invocation.
    """
    seg = _segment("a", "b", "c", "d")  # all 4 confidently unsupported
    primary, ctx1, ctx2 = _image("red"), _image("blue"), _image("green")
    vlm = _FakeVLM(lambda _img, claim: _unsupported(claim))

    audit_segment(seg, [primary, ctx1, ctx2], vlm)

    # 1 batch on primary + 1 batch per context frame = 3 total batches.
    assert len(vlm.batch_calls) == 3
    # No individual verify_claim invocations — everything is batched.
    assert vlm.single_calls == []
    # Each context batch carries all 4 still-pending claims.
    primary_batch, *context_batches = vlm.batch_calls
    assert len(primary_batch[1]) == 4
    for _, claims in context_batches:
        assert len(claims) == 4


def test_context_rescue_short_circuits_on_first_supporting_frame() -> None:
    seg = _segment("woman")
    primary = _image("red")
    ctx1, ctx2 = _image("blue"), _image("green")

    def responder(img: Image.Image, claim: str) -> VerificationResult:
        if img is primary:
            return _unsupported(claim)
        if img is ctx1:
            return _supported(claim)
        return _supported(claim)  # ctx2 — should never be reached

    vlm = _FakeVLM(responder)
    audit_segment(seg, [primary, ctx1, ctx2], vlm)

    ids_queried = {id(img) for img, _ in vlm.calls}
    assert id(primary) in ids_queried
    assert id(ctx1) in ids_queried
    assert id(ctx2) not in ids_queried  # short-circuited


def test_mixed_claims_yield_partial_hallucination() -> None:
    # 3 supported + 2 confidently-unsupported (no rescue) = 3/5 = 0.6
    # 0.6 is between partial_threshold=0.4 and clean_threshold=0.8 → partial.
    seg = _segment("a", "b", "c", "d", "e")
    supported_set = {"a", "b", "c"}

    def responder(_img: Image.Image, claim: str) -> VerificationResult:
        return _supported(claim) if claim in supported_set else _unsupported(claim)

    vlm = _FakeVLM(responder)
    result = audit_segment(seg, [_image()], vlm)

    assert result.grounding_score == pytest.approx(0.6)
    assert result.verdict == "partial_hallucination"
    assert result.hallucination_count == 2


def test_verdict_thresholds_are_tunable() -> None:
    """DD-12: thresholds are defaults, not asserted; CLI surfaces them."""
    seg = _segment("a", "b", "c", "d", "e")
    supported_set = {"a", "b", "c", "d"}  # 4/5 = 0.8

    def responder(_img: Image.Image, claim: str) -> VerificationResult:
        return _supported(claim) if claim in supported_set else _unsupported(claim)

    vlm = _FakeVLM(responder)

    default = audit_segment(seg, [_image()], vlm)
    assert default.verdict == "clean"  # 0.8 >= default clean_threshold=0.8

    stricter = audit_segment(seg, [_image()], vlm, clean_threshold=0.9)
    assert stricter.verdict == "partial_hallucination"  # 0.8 < 0.9


def test_empty_claims_segment_is_vacuously_clean() -> None:
    seg = DescriptionSegment(timestamp_start=0.0, timestamp_end=1.0, description="ø", claims=[])
    vlm = _FakeVLM(lambda _img, claim: _supported(claim))

    result = audit_segment(seg, [_image()], vlm)

    assert result.verdict == "clean"
    assert result.grounding_score == 1.0
    assert result.hallucination_count == 0
    assert result.claim_results == []
    assert vlm.calls == []  # VLM not consulted at all


def test_empty_frames_raises_value_error() -> None:
    seg = _segment("x")
    vlm = _FakeVLM(lambda _img, claim: _supported(claim))

    with pytest.raises(ValueError, match="at least one frame"):
        audit_segment(seg, [], vlm)


def test_result_round_trips_through_json() -> None:
    seg = _segment("woman")
    vlm = _FakeVLM(lambda _img, claim: _supported(claim))

    result = audit_segment(seg, [_image()], vlm)
    restored = SegmentAuditResult.model_validate_json(result.model_dump_json())

    assert restored == result
