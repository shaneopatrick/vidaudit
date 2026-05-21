"""Audit one description segment against its sampled frames.

This is the orchestration of DD-1 (claims-based verification): the parser
decomposes a description into claims, the frame sampler delivers a primary
frame plus context frames spanning the segment (DD-9), and this module wires
each claim through the VLM, applying DD-7 confidence semantics and DD-12
eval-derived thresholds.

The audit receives frames already sampled — the caller (CLI orchestration)
is responsible for resolving DD-9's missing-end fallback chain and
constructing the ``[primary, *context]`` frame list, since that decision
needs the ordered segment list and the video, neither of which are in this
module's contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from vidaudit.description_parser import Claim, DescriptionSegment
from vidaudit.vlm.base import VerificationResult

if TYPE_CHECKING:
    from PIL import Image

    from vidaudit.vlm.base import VLMBackend


SegmentVerdict = Literal["clean", "partial_hallucination", "full_hallucination"]


class ClaimResult(BaseModel):
    """A single claim's verdict after the audit (primary ± context check)."""

    claim: Claim
    verification: VerificationResult
    flagged: bool


class SegmentAuditResult(BaseModel):
    """The audit outcome for one description segment."""

    segment: DescriptionSegment
    claim_results: list[ClaimResult]
    grounding_score: float
    hallucination_count: int
    verdict: SegmentVerdict


def audit_segment(
    segment: DescriptionSegment,
    frames: list[Image.Image],
    vlm: VLMBackend,
    confidence_threshold: float = 0.3,
    clean_threshold: float = 0.8,
    partial_threshold: float = 0.4,
) -> SegmentAuditResult:
    """Audit one segment's claims against its sampled frames.

    Verification proceeds in two passes:

    1. Batch-verify every claim against the primary frame.
    2. For each claim that came back as confidently ``unsupported`` on the
       primary frame, check the context frames; a confident ``supported``
       from any context frame rescues the claim and overrides the result.

    Per DD-7, ``confidence`` is the VLM's confidence in its verdict. A
    low-confidence ``unsupported`` is treated as ``uncertain`` — neither
    counted toward grounding nor flagged as a hallucination — and is NOT
    escalated to context frames.

    Args:
        segment: The decomposed segment whose claims will be checked.
        frames: ``[primary, *context]`` for this segment. Primary at index 0
            per the frame_sampler contract.
        vlm: The VLM backend.
        confidence_threshold: Minimum verdict-confidence (DD-7) for an
            ``unsupported`` verdict to be taken seriously enough to flag and
            for a ``supported`` context frame to rescue a claim. Default 0.3.
        clean_threshold: Lower bound on ``grounding_score`` for the
            ``"clean"`` verdict (DD-12 default — surface as a CLI flag and
            calibrate via the eval threshold sweep). Default 0.8.
        partial_threshold: Lower bound on ``grounding_score`` for
            ``"partial_hallucination"``; below this is
            ``"full_hallucination"``. Default 0.4.

    Returns:
        The full segment audit result.

    Raises:
        ValueError: If ``frames`` is empty.
    """
    if not frames:
        raise ValueError(
            f"audit_segment requires at least one frame; got none for segment "
            f"at t={segment.timestamp_start:.2f}s"
        )

    primary, *context = frames

    if not segment.claims:
        # No extractable claims (description_parser edge case): nothing to
        # falsify, so the segment is vacuously clean. Eval should filter these
        # out separately so they don't inflate the aggregate grounding score.
        return SegmentAuditResult(
            segment=segment,
            claim_results=[],
            grounding_score=1.0,
            hallucination_count=0,
            verdict="clean",
        )

    claim_texts = [c.text for c in segment.claims]
    primary_results = vlm.verify_batch(primary, claim_texts)

    verifications = list(primary_results)
    flagged = [False] * len(segment.claims)

    # Claims that came back confidently "unsupported" on the primary frame —
    # candidates for context-frame rescue. Per DD-7, a LOW-confidence
    # "unsupported" is uncertain, not a hallucination, and is NOT escalated.
    pending: set[int] = {
        i
        for i, v in enumerate(primary_results)
        if v.verdict == "unsupported" and v.confidence > confidence_threshold
    }

    # Batch all pending claims against each context frame in turn (DD-6) —
    # one VLM call per context frame, not one per (claim, frame) pair. We
    # short-circuit at the frame level once every pending claim has been
    # rescued (still better than re-checking already-rescued claims).
    for ctx_frame in context:
        if not pending:
            break
        pending_indices = sorted(pending)
        pending_texts = [segment.claims[i].text for i in pending_indices]
        ctx_results = vlm.verify_batch(ctx_frame, pending_texts)
        for local_idx, idx in enumerate(pending_indices):
            v = ctx_results[local_idx]
            if v.verdict == "supported" and v.confidence > confidence_threshold:
                verifications[idx] = v
                pending.discard(idx)

    # Anything still pending was not rescued by any context frame → flag.
    for idx in pending:
        flagged[idx] = True

    claim_results = [
        ClaimResult(claim=c, verification=v, flagged=f)
        for c, v, f in zip(segment.claims, verifications, flagged, strict=True)
    ]

    total = len(claim_results)
    supported = sum(1 for cr in claim_results if cr.verification.verdict == "supported")
    grounding_score = supported / total

    if grounding_score >= clean_threshold:
        verdict: SegmentVerdict = "clean"
    elif grounding_score >= partial_threshold:
        verdict = "partial_hallucination"
    else:
        verdict = "full_hallucination"

    return SegmentAuditResult(
        segment=segment,
        claim_results=claim_results,
        grounding_score=grounding_score,
        hallucination_count=sum(1 for f in flagged if f),
        verdict=verdict,
    )
