"""Weak captioners for harvesting *real* hallucinations (DD-13).

The real-hallucination subset is built by running a captioner over sampled
frames and keeping its naturally-generated descriptions — these resemble the
error distribution of a production system far better than synthetic mutations
do, which is what gives the eval face validity (DD-13).

A :class:`Captioner` is just ``(image) -> str``. Two concrete ones ship here:

* :class:`GeminiCaptioner` — no-GPU dev path, reuses the ``google-genai`` SDK.
* :func:`qwen_captioner` — adapts an existing :class:`QwenRunner` (the same
  ``(image, prompt) -> str`` runner the Qwen backend uses), so the canonical
  open-weight backend doubles as the captioner with no extra model load.

Both are deliberately thin: the eval injects whichever one matches the
environment, and tests inject a fake.
"""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from google import genai
    from PIL import Image

    from vidaudit.vlm.qwen_vl import QwenRunner


DEFAULT_CAPTION_PROMPT = (
    "Describe what is visible in this single video frame in one or two "
    "sentences. Mention the main objects, people, and setting. Do not "
    "speculate about events outside the frame."
)


class Captioner(Protocol):
    """Turns a frame into a free-text caption."""

    def __call__(self, image: Image.Image) -> str: ...


def qwen_captioner(runner: QwenRunner, prompt: str = DEFAULT_CAPTION_PROMPT) -> Captioner:
    """Adapt a :class:`QwenRunner` into a :class:`Captioner`.

    The Qwen backend's runner is already ``(image, prompt) -> str``, so a
    caption is just that runner invoked with a captioning prompt — no second
    model load.
    """

    def _caption(image: Image.Image) -> str:
        return runner(image, prompt).strip()

    return _caption


class GeminiCaptioner:
    """Free-text frame captioner backed by ``google-genai``.

    Unlike :class:`~vidaudit.vlm.gemini.GeminiBackend` this asks for prose,
    not a structured verdict — it exists to *generate* descriptions to audit,
    not to verify claims.

    Args:
        model: Gemini model id.
        prompt: Captioning instruction.
        client: Injected SDK client (tests pass a mock; never hits the real
            API in CI per CLAUDE.md §6).
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        prompt: str = DEFAULT_CAPTION_PROMPT,
        client: genai.Client | None = None,
    ) -> None:
        if client is None:
            import os

            from google import genai

            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("Set GEMINI_API_KEY environment variable (or pass client=).")
            client = genai.Client(api_key=key)
        self._client = client
        self._model = model
        self._prompt = prompt

    def __call__(self, image: Image.Image) -> str:
        from google.genai import types

        buf = BytesIO()
        image.save(buf, format="PNG")
        response = self._client.models.generate_content(
            model=self._model,
            contents=[  # type: ignore[arg-type]  # list[Part] is fine; SDK union is invariant
                types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
                types.Part.from_text(text=self._prompt),
            ],
        )
        text = response.text
        return text.strip() if text else ""
