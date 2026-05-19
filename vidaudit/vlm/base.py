"""VLM backend interface and the verification-result model.

A backend takes a frame + a claim and decides whether the claim is visually
supported. Backends are pluggable behind :class:`VLMBackend` (DESIGN.md DD-3)
so the eval can compare implementations head-to-head (DD-13, DD-16).

The default ``verify_batch`` loops over ``verify_claim``; backends with a true
batched API (e.g. Gemini) override it for the quota win (DD-6).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from PIL import Image

Verdict = Literal["supported", "unsupported", "uncertain"]


class VerificationResult(BaseModel):
    """One verifier's verdict on one claim against one frame.

    ``confidence`` follows DD-7: it is the VLM's confidence in the verdict it
    just gave (1.0 = certain, 0.0 = pure guess), NOT the probability the claim
    is true.
    """

    claim: str
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = ""


class VLMBackend(ABC):
    """Pluggable VLM backend interface (DD-3).

    Concrete backends MUST set ``model_id`` to a stable, pinned identifier
    (e.g. ``gemini-2.5-flash``, ``Qwen/Qwen2.5-VL-3B-Instruct@<sha>``). The
    identifier ends up in the verification cache key (DD-11) and the report
    metadata, so reproducibility depends on it being precise.
    """

    model_id: str

    @abstractmethod
    def verify_claim(self, image: Image.Image, claim: str) -> VerificationResult:
        """Check whether a single claim is visually supported by the frame."""

    def verify_batch(self, image: Image.Image, claims: list[str]) -> list[VerificationResult]:
        """Verify multiple claims against the same frame.

        Default loops over :meth:`verify_claim`; backends with a real batched
        API override this. Order of the returned list matches ``claims``.
        """
        return [self.verify_claim(image, c) for c in claims]
