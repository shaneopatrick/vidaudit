"""Qwen2.5-VL-3B-Instruct backend — the canonical eval backend.

Open-weight verifier via ``transformers``. Reproducibility-anchored: a pinned
model revision (commit SHA) means the cache key and reported metrics are stable
forever — no risk of silent provider-side behavior change.

Unlike Gemini there is no native ``response_schema`` plumbing, so JSON shape
is requested in the prompt and parsed/validated against the same Pydantic
models on the way out. A best-effort regex fallback handles the case where
the model wraps JSON in markdown fences.

Heavy deps (``transformers``, ``torch``, ``accelerate``) live behind the
``[qwen]`` extra and are imported lazily inside :class:`_HFRunner` — importing
this module on a no-GPU machine is fine; constructing :class:`QwenVLBackend`
without a stub ``runner=`` is what pulls them in.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from vidaudit.vlm._cache import cache_get, cache_key, cache_put
from vidaudit.vlm.base import VerificationResult, VLMBackend

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
# Pin the revision when you ship eval numbers, for reproducibility. Left as
# None in the default so dev work doesn't fail closed; the Colab eval driver
# and CLI both surface ``--qwen-revision`` so reported metrics can be anchored
# to an exact commit SHA.
DEFAULT_REVISION: str | None = None
DEFAULT_MAX_NEW_TOKENS = 256


_CONFIDENCE_DESCRIPTION = (
    "Your confidence in the verdict (1.0 = certain, 0.0 = pure guess). This "
    "is NOT the probability that the claim is true — it is your confidence "
    "in the verdict label you just gave."
)


class _QwenVerdict(BaseModel):
    """Single-claim response schema (mirrors :class:`_GeminiVerdict`)."""

    verdict: Literal["supported", "unsupported", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0, description=_CONFIDENCE_DESCRIPTION)
    evidence: str = ""


class _QwenBatchVerdict(BaseModel):
    """One item in a batched Qwen response — echoes the claim text for matching."""

    claim: str
    verdict: Literal["supported", "unsupported", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0, description=_CONFIDENCE_DESCRIPTION)
    evidence: str = ""


_SINGLE_PROMPT = (
    "You are a video frame auditor. You are shown one frame from a video and "
    "one claim about what appears in that frame. Decide whether the claim is "
    "visually supported by the frame.\n\n"
    'Claim: "{claim}"\n\n'
    "Respond with ONLY a JSON object, no markdown fences, no prose:\n"
    '{{"verdict": "supported"|"unsupported"|"uncertain", '
    '"confidence": <0.0-1.0>, "evidence": "<one short sentence>"}}\n\n'
    "`confidence` is your confidence in the verdict label, not the probability "
    "the claim is true."
)


def _batch_prompt(claims: list[str]) -> str:
    numbered = "\n".join(f'{i + 1}. "{c}"' for i, c in enumerate(claims))
    return (
        "You are a video frame auditor. You are shown one frame from a video "
        "and a list of claims about what appears in that frame. For EACH "
        "claim, decide whether it is visually supported by the frame. Echo "
        "each claim back verbatim; return one verdict per claim, in input "
        "order.\n\n"
        "Claims:\n" + numbered + "\n\n"
        "Respond with ONLY a JSON array, no markdown fences, no prose:\n"
        '[{"claim": "<echo>", "verdict": "supported"|"unsupported"|"uncertain", '
        '"confidence": <0.0-1.0>, "evidence": "<one short sentence>"}, ...]\n\n'
        "`confidence` is your confidence in the verdict label, not the "
        "probability the claim is true."
    )


class QwenRunner(Protocol):
    """Callable that turns ``(image, prompt)`` into a model text response.

    Defined so tests can inject a stub without importing ``transformers``;
    the real implementation is :class:`_HFRunner`.
    """

    def __call__(self, image: Image.Image, prompt: str) -> str: ...


class QwenVLBackend(VLMBackend):
    """Qwen2.5-VL-3B-Instruct backend.

    Args:
        model: HF model id. Defaults to ``Qwen/Qwen2.5-VL-3B-Instruct``.
        revision: Pinned commit SHA — strongly recommended for reproducible
            eval runs. When set, the cache key and the report's ``model_id``
            both include it, so reruns at a later SHA don't silently reuse
            stale verdicts.
        device: ``"cuda"``, ``"cpu"``, ``"auto"``, or ``None`` (auto). Passed
            as ``device_map`` to ``from_pretrained``.
        load_in_4bit: Enable bitsandbytes 4-bit quantization (requires
            ``bitsandbytes`` extra). Cuts VRAM ~7 GB → ~4 GB on the 3B
            checkpoint — useful for consumer GPUs.
        max_new_tokens: Token budget per call. 256 is comfortably enough for
            a batched JSON array of ~10 short verdicts.
        runner: Optional injected runner (for tests). When provided, the HF
            model + processor are NOT loaded — handy because constructing a
            real runner pulls in torch + a multi-GB model download.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        revision: str | None = DEFAULT_REVISION,
        device: str | None = None,
        load_in_4bit: bool = False,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        runner: QwenRunner | None = None,
    ) -> None:
        self._runner: QwenRunner = runner or _HFRunner(
            model=model,
            revision=revision,
            device=device,
            load_in_4bit=load_in_4bit,
            max_new_tokens=max_new_tokens,
        )
        # Cache key + report metadata pin the revision when available so a
        # later checkpoint SHA can't silently reuse the old cached verdicts.
        self.model_id = f"{model}@{revision}" if revision else model

    def verify_claim(self, image: Image.Image, claim: str) -> VerificationResult:
        key = cache_key(image, claim, self.model_id)
        cached = cache_get(key)
        if cached is not None:
            return cached

        text = self._runner(image, _SINGLE_PROMPT.format(claim=claim))
        parsed = _parse_single(text)
        if parsed is None:
            result = _uncertain(claim, "VLM returned no parseable JSON.")
        else:
            result = VerificationResult(claim=claim, **parsed.model_dump())
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
            text = self._runner(image, _batch_prompt(uncached))
            by_text = _parse_batch(text, uncached)
            for c in uncached:
                item = by_text.get(c)
                if item is None:
                    result = _uncertain(c, "VLM omitted this claim from the batched response.")
                else:
                    result = VerificationResult(
                        claim=c,
                        verdict=item.verdict,
                        confidence=item.confidence,
                        evidence=item.evidence,
                    )
                cache_put(cache_key(image, c, self.model_id), result)
                by_claim[c] = result

        return [by_claim[c] for c in claims]


# ---- text -> structured-response parsing ---------------------------------


def _parse_single(text: str) -> _QwenVerdict | None:
    """Parse the model's text into a :class:`_QwenVerdict`, or ``None``."""
    payload = _extract_json_payload(text)
    if payload is None:
        return None
    try:
        return _QwenVerdict.model_validate(payload)
    except ValidationError as exc:
        logger.warning("Qwen single response failed validation: %s", exc)
        return None


def _parse_batch(text: str, requested: list[str]) -> dict[str, _QwenBatchVerdict]:
    """Parse a batched response into ``{claim_text: verdict}``.

    Falls back to positional alignment if lengths match but claim texts don't
    — Qwen sometimes paraphrases ("The Eiffel Tower" → "Eiffel Tower").
    """
    payload = _extract_json_payload(text)
    if not isinstance(payload, list):
        return {}
    items: list[_QwenBatchVerdict] = []
    for raw in payload:
        try:
            items.append(_QwenBatchVerdict.model_validate(raw))
        except ValidationError as exc:
            logger.warning("Qwen batch item failed validation: %s — raw=%r", exc, raw)
            continue

    by_text: dict[str, _QwenBatchVerdict] = {item.claim: item for item in items}
    if len(items) == len(requested):
        for c, item in zip(requested, items, strict=False):
            by_text.setdefault(c, item)
    return by_text


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json_payload(text: str) -> Any:
    """Best-effort JSON extraction from a free-text model response.

    Tries in order: parse as-is, strip a single ```json fence, locate the
    outermost ``{...}`` or ``[...]`` block. Returns ``None`` if nothing
    parses — caller treats that as ``uncertain`` (last-resort fallback).
    """
    if not text:
        return None
    candidate = text.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    fence = _FENCE_RE.search(candidate)
    if fence:
        inner = fence.group(1).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass

    # Locate the outermost JSON value by bracket matching. We try array
    # before object so a batched-response list isn't mis-parsed as the first
    # element only.
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = candidate.find(open_ch)
        if start == -1:
            continue
        end = candidate.rfind(close_ch)
        if end <= start:
            continue
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            continue

    return None


def _uncertain(claim: str, reason: str) -> VerificationResult:
    return VerificationResult(claim=claim, verdict="uncertain", confidence=0.0, evidence=reason)


# ---- HuggingFace runner (heavy deps live here) ---------------------------


class _HFRunner:
    """Real ``(image, prompt) -> text`` runner backed by ``transformers``.

    Loads the model + processor at construction. Greedy decoding
    (``do_sample=False``) for determinism. Lazy-imports torch and
    transformers so importing :mod:`vidaudit.vlm.qwen_vl` itself doesn't
    require the ``[qwen]`` extra — only instantiating a real backend does.
    """

    def __init__(
        self,
        model: str,
        revision: str | None,
        device: str | None,
        load_in_4bit: bool,
        max_new_tokens: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover — exercised on no-GPU systems
            raise RuntimeError(
                "Qwen backend requires the [qwen] extra. Install with:\n"
                "  uv sync --extra qwen\n"
                "or\n"
                "  pip install 'vidaudit[qwen]'"
            ) from exc

        kwargs: dict[str, object] = {"torch_dtype": torch.float16}
        if revision is not None:
            kwargs["revision"] = revision
        kwargs["device_map"] = device if device is not None else "auto"

        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "load_in_4bit=True requires bitsandbytes. Install with:\n"
                    "  pip install bitsandbytes"
                ) from exc
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            # device_map is incompatible with explicit device strings under
            # quantization; let accelerate decide.
            kwargs["device_map"] = "auto"

        logger.info("Loading Qwen2.5-VL model %s (revision=%s)…", model, revision)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **kwargs)
        self._processor = AutoProcessor.from_pretrained(
            model, revision=revision, trust_remote_code=False
        )
        self._max_new_tokens = max_new_tokens
        # Cached for speed; tokenize once.
        self._torch = torch

    def __call__(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[chat_text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with self._torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )

        prompt_len = inputs.input_ids.shape[1]
        generated = output_ids[:, prompt_len:]
        decoded: list[str] = self._processor.batch_decode(generated, skip_special_tokens=True)
        return decoded[0] if decoded else ""


# Honoured by callers (CLI, eval driver) that want to override the model id
# via env without surfacing a separate flag — e.g. swapping in a 7B variant
# for a scaling comparison.
def model_from_env(default: str = DEFAULT_MODEL) -> str:
    return os.environ.get("VIDAUDIT_QWEN_MODEL", default)
