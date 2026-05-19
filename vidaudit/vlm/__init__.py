"""Pluggable VLM backends for claim verification."""

from __future__ import annotations

from vidaudit.vlm.base import VerificationResult, VLMBackend

__all__ = ["VLMBackend", "VerificationResult"]
