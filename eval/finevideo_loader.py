"""FineVideo loader + labeled mutation dataset.

Three things live here:

1. **Loading** real chapter descriptions from the FineVideo dataset.
2. **Synthetic mutations** — deterministic, *plausible* corruptions of clean
   descriptions (object swap, attribute change, entity injection). Plausible,
   context-consistent swaps via hand-curated tables — random swaps are
   trivially detectable and would inflate the metrics. Curated tables are also
   fully auditable: a reviewer can see every possible swap.
3. **Real hallucinations** — pairs of (ground-truth, captioner-generated)
   descriptions; the captioner's natural errors are the realistic subset that
   gives the eval face validity. Captioning is injected
   (:mod:`eval.captioner`) so this module stays GPU/network-free for tests.

Every produced :class:`EvalSample` carries a ``source`` (``synthetic`` /
``real``) and, for synthetic mutations, the exact span that was corrupted —
that span is the ground-truth "should be flagged" claim the eval scores
against. The two subsets are reported separately, never averaged.
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, TypeAdapter

from vidaudit.description_parser import _load_spacy

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from PIL import Image

    from eval.captioner import Captioner

logger = logging.getLogger(__name__)

FINEVIDEO_DATASET = "HuggingFaceFV/finevideo"


class MutationType(str, Enum):
    object_swap = "object_swap"
    attribute_mutation = "attribute_mutation"
    entity_injection = "entity_injection"


# Object head-noun -> plausible, context-consistent replacements. Swaps stay
# within a rough semantic neighbourhood (a likely co-occurring object) so the
# corruption isn't trivially detectable. Keyed by singular lemma.
_PLAUSIBLE_SWAPS: dict[str, list[str]] = {
    "dog": ["cat", "fox"],
    "cat": ["dog", "rabbit"],
    "car": ["truck", "van"],
    "truck": ["car", "bus"],
    "bicycle": ["motorcycle", "scooter"],
    "man": ["woman", "boy"],
    "woman": ["man", "girl"],
    "child": ["adult", "teenager"],
    "boat": ["ship", "canoe"],
    "tree": ["bush", "palm"],
    "guitar": ["violin", "banjo"],
    "coffee": ["tea", "juice"],
    "phone": ["tablet", "camera"],
    "book": ["magazine", "notebook"],
    "horse": ["cow", "donkey"],
    "bird": ["butterfly", "bat"],
    "mountain": ["hill", "cliff"],
    "river": ["lake", "stream"],
    "building": ["house", "tower"],
    "table": ["desk", "counter"],
}

# Attribute (adjective) -> plausible alternative. Colours and sizes — a
# colour mutation is the canonical "looks fine, factually wrong" hallucination.
_ATTRIBUTE_SWAPS: dict[str, list[str]] = {
    "red": ["blue", "green"],
    "blue": ["red", "yellow"],
    "green": ["red", "orange"],
    "yellow": ["purple", "blue"],
    "black": ["white", "grey"],
    "white": ["black", "brown"],
    "orange": ["pink", "green"],
    "purple": ["yellow", "green"],
    "brown": ["grey", "black"],
    "small": ["large", "tiny"],
    "large": ["small", "huge"],
    "tall": ["short", "narrow"],
    "short": ["tall", "long"],
    "young": ["old", "elderly"],
    "old": ["young", "new"],
}

# Well-known named entities to inject. Picked to be obviously absent from
# ordinary footage, so an injection is a genuine hallucination to detect.
_INJECTABLE_ENTITIES: list[str] = [
    "the Eiffel Tower",
    "the Statue of Liberty",
    "Big Ben",
    "the Golden Gate Bridge",
    "Mount Fuji",
    "the Colosseum",
]


class FineVideoChapter(BaseModel):
    """One time-coded chapter/scene with its ground-truth description."""

    video_id: str
    timestamp_start: float
    timestamp_end: float | None = None
    description: str


class EvalSample(BaseModel):
    """A labeled clean-vs-mutated pair for the eval.

    For synthetic mutations, ``mutated_span`` is the false phrase introduced
    (the claim the auditor *should* flag) and ``original_span`` is what it
    replaced. A clean control has ``mutation_type=None`` and no spans. A real
    sample has ``source="real"``, no mutation type, and unlabeled spans — its
    hallucinated claims are hand-labeled during the eval run.
    """

    video_id: str
    timestamp_start: float
    timestamp_end: float | None = None
    clean_description: str
    mutated_description: str
    mutation_type: MutationType | None = None
    original_span: str | None = None
    mutated_span: str | None = None
    source: Literal["synthetic", "real"]
    # Hand-label for the real subset: True if the caption contains a
    # hallucination, False if clean, None if not yet labeled. Synthetic samples
    # derive their ground truth from ``mutation_type`` and ignore this.
    real_is_hallucinated: bool | None = None


_SAMPLES_ADAPTER = TypeAdapter(list[EvalSample])


# ---- synthetic mutations --------------------------------------------------


def _replace_first_word(text: str, old: str, new: str) -> str:
    """Replace the first whole-word, case-insensitive occurrence of ``old``."""
    pattern = re.compile(rf"\b{re.escape(old)}\b", re.IGNORECASE)
    return pattern.sub(new, text, count=1)


def _clean_sample(chapter: FineVideoChapter) -> EvalSample:
    """A no-mutation control — used to measure false positives."""
    return EvalSample(
        video_id=chapter.video_id,
        timestamp_start=chapter.timestamp_start,
        timestamp_end=chapter.timestamp_end,
        clean_description=chapter.description,
        mutated_description=chapter.description,
        mutation_type=None,
        source="synthetic",
    )


def _mutate_object_swap(chapter: FineVideoChapter) -> EvalSample | None:
    """Swap one head noun for a plausible co-occurring object."""
    doc = _load_spacy()(chapter.description)
    for chunk in doc.noun_chunks:
        head = chunk.root
        lemma = head.lemma_.lower()
        if lemma in _PLAUSIBLE_SWAPS:
            replacement = _PLAUSIBLE_SWAPS[lemma][0]
            mutated = _replace_first_word(chapter.description, head.text, replacement)
            if mutated == chapter.description:
                continue
            return EvalSample(
                video_id=chapter.video_id,
                timestamp_start=chapter.timestamp_start,
                timestamp_end=chapter.timestamp_end,
                clean_description=chapter.description,
                mutated_description=mutated,
                mutation_type=MutationType.object_swap,
                original_span=head.text.lower(),
                mutated_span=replacement,
                source="synthetic",
            )
    return None


def _mutate_attribute(chapter: FineVideoChapter) -> EvalSample | None:
    """Change one adjective (colour/size) to a plausible alternative."""
    doc = _load_spacy()(chapter.description)
    for token in doc:
        lemma = token.lemma_.lower()
        if token.pos_ == "ADJ" and lemma in _ATTRIBUTE_SWAPS:
            replacement = _ATTRIBUTE_SWAPS[lemma][0]
            mutated = _replace_first_word(chapter.description, token.text, replacement)
            if mutated == chapter.description:
                continue
            # The auditor extracts "<adj> <noun>" as one claim, so label the
            # span as the full mutated phrase when the adjective modifies a noun.
            noun = token.head.text if token.head.pos_ in {"NOUN", "PROPN"} else ""
            mutated_span = f"{replacement} {noun}".strip()
            original_span = f"{token.text.lower()} {noun}".strip()
            return EvalSample(
                video_id=chapter.video_id,
                timestamp_start=chapter.timestamp_start,
                timestamp_end=chapter.timestamp_end,
                clean_description=chapter.description,
                mutated_description=mutated,
                mutation_type=MutationType.attribute_mutation,
                original_span=original_span,
                mutated_span=mutated_span,
                source="synthetic",
            )
    return None


def _mutate_entity_injection(chapter: FineVideoChapter) -> EvalSample | None:
    """Append a well-known landmark that is not already mentioned."""
    lowered = chapter.description.lower()
    for entity in _INJECTABLE_ENTITIES:
        if entity.lower() in lowered:
            continue
        base = chapter.description.rstrip()
        connector = " " if base.endswith((".", "!", "?")) else ", with "
        if connector == " ":
            mutated = f"{base} {entity} is visible in the background."
        else:
            mutated = f"{base}{connector}{entity} in the background."
        return EvalSample(
            video_id=chapter.video_id,
            timestamp_start=chapter.timestamp_start,
            timestamp_end=chapter.timestamp_end,
            clean_description=chapter.description,
            mutated_description=mutated,
            mutation_type=MutationType.entity_injection,
            original_span=None,
            mutated_span=entity,
            source="synthetic",
        )
    return None


def make_synthetic_samples(
    chapter: FineVideoChapter, *, include_clean: bool = True
) -> list[EvalSample]:
    """All applicable synthetic samples for one chapter.

    Emits a clean control (for false-positive measurement) plus one sample per
    mutation type that applies to the description. A mutation that can't be
    applied (no swappable noun, no colour adjective, etc.) is skipped — not
    forced — so the dataset never contains degenerate "mutations" identical to
    the clean text.
    """
    samples: list[EvalSample] = []
    if include_clean:
        samples.append(_clean_sample(chapter))
    for mutator in (_mutate_object_swap, _mutate_attribute, _mutate_entity_injection):
        sample = mutator(chapter)
        if sample is not None:
            samples.append(sample)
    return samples


# ---- real (captioner-harvested) subset ------------------------------------


def build_real_samples(
    chapters: Iterable[FineVideoChapter],
    captioner: Captioner,
    frame_for: Callable[[FineVideoChapter], Image.Image | None],
) -> list[EvalSample]:
    """Harvest real hallucinations by captioning each chapter's frame.

    For each chapter a representative frame is fetched via ``frame_for`` and
    passed to ``captioner``; the generated caption becomes ``mutated`` and the
    FineVideo ground-truth becomes ``clean``. Hallucinated claims in the
    caption are hand-labeled later in the eval run — this builder does not
    auto-label them.

    ``frame_for`` returning ``None`` (e.g. timestamp past the clip) skips the
    chapter with a warning.
    """
    samples: list[EvalSample] = []
    for chapter in chapters:
        frame = frame_for(chapter)
        if frame is None:
            logger.warning(
                "No frame for %s @ %.1fs — skipping real sample.",
                chapter.video_id,
                chapter.timestamp_start,
            )
            continue
        caption = captioner(frame)
        if not caption.strip():
            logger.warning("Empty caption for %s — skipping.", chapter.video_id)
            continue
        samples.append(
            EvalSample(
                video_id=chapter.video_id,
                timestamp_start=chapter.timestamp_start,
                timestamp_end=chapter.timestamp_end,
                clean_description=chapter.description,
                mutated_description=caption,
                mutation_type=None,
                source="real",
            )
        )
    return samples


# ---- FineVideo loading ----------------------------------------------------


def chapters_from_row(row: dict, video_id: str) -> list[FineVideoChapter]:
    """Extract chapters from one FineVideo dataset row.

    FineVideo stores rich JSON metadata per video under ``content_metadata``.
    Each scene's times are nested timecode strings and there is no flat
    per-scene ``description`` — the visual content lives in ``activities``::

        row["json"]["content_metadata"]["scenes"] = [
            {
                "title": "Introductory Scenes",
                "timestamps": {
                    "start_timestamp": "00:00:00.000",
                    "end_timestamp": "00:00:28.779",
                },
                "activities": [{"description": "...", "timestamp": {...}}, ...],
            },
            ...
        ]

    Per scene the description is taken from (in order) a flat ``description`` /
    ``text`` field, the joined ``activities`` descriptions, or the ``title``.
    Times accept numeric seconds, plain numeric strings, or ``HH:MM:SS.mmm``
    timecodes — and either flat keys or the nested ``timestamps`` block.
    ``row["json"]`` may be a dict or a JSON-encoded string. Scenes with no
    usable description or start time are skipped. Deliberately tolerant of
    schema drift across dataset releases.
    """
    meta = row.get("json")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            logger.warning("Could not decode FineVideo json for %s", video_id)
            return []
    if not isinstance(meta, dict):
        return []

    content = meta.get("content_metadata", meta)
    scenes = content.get("scenes") or content.get("chapters") or []
    if not isinstance(scenes, list):
        return []

    chapters: list[FineVideoChapter] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        description = (
            scene.get("description")
            or scene.get("text")
            or _join_activities(scene.get("activities"))
            or scene.get("title")
        )
        start = _scene_time(scene, ("start_time", "timestamp_start", "start"), "start_timestamp")
        if not description or start is None:
            continue
        end = _scene_time(scene, ("end_time", "timestamp_end", "end"), "end_timestamp")
        chapters.append(
            FineVideoChapter(
                video_id=video_id,
                timestamp_start=start,
                timestamp_end=end,
                description=str(description).strip(),
            )
        )
    return chapters


def _join_activities(activities: object) -> str:
    if not isinstance(activities, list):
        return ""
    parts = [str(a.get("description", "")) for a in activities if isinstance(a, dict)]
    return " ".join(p for p in parts if p).strip()


def _parse_timecode(value: object) -> float | None:
    """Parse seconds from a number, numeric string, or ``HH:MM:SS.mmm`` timecode."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    seconds = 0.0
    for part in text.split(":"):
        try:
            seconds = seconds * 60 + float(part)
        except ValueError:
            return None
    return seconds


def _scene_time(scene: dict, flat_keys: tuple[str, ...], nested_key: str) -> float | None:
    """Resolve a scene time from flat keys or the nested ``timestamps`` block."""
    for key in flat_keys:
        if key in scene:
            parsed = _parse_timecode(scene[key])
            if parsed is not None:
                return parsed
    timestamps = scene.get("timestamps")
    if isinstance(timestamps, dict):
        return _parse_timecode(timestamps.get(nested_key))
    return None


def load_finevideo(
    split: str = "train",
    limit: int = 10,
    streaming: bool = True,
) -> list[FineVideoChapter]:
    """Load chapter descriptions from FineVideo (requires the ``eval`` extra).

    Streams ``limit`` videos and flattens their scenes into chapters. FineVideo
    is large and gated behind a HF login; this is a thin convenience used in
    the Colab eval — it is not exercised in unit tests (the pure
    :func:`chapters_from_row` is).

    Raises:
        RuntimeError: If the ``datasets`` library is not installed.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "FineVideo loading requires the [eval] extra: uv sync --extra eval"
        ) from exc

    ds = hf_load_dataset(FINEVIDEO_DATASET, split=split, streaming=streaming)
    chapters: list[FineVideoChapter] = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        video_id = str(row.get("id", row.get("video_id", f"video_{i}")))
        chapters.extend(chapters_from_row(row, video_id))
    return chapters


# ---- dataset I/O ----------------------------------------------------------


def save_dataset(samples: list[EvalSample], path: Path, *, indent: int = 2) -> None:
    """Write eval samples to JSON (creates parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_SAMPLES_ADAPTER.dump_json(samples, indent=indent))


def load_dataset(path: Path) -> list[EvalSample]:
    """Load eval samples from a JSON file written by :func:`save_dataset`."""
    return _SAMPLES_ADAPTER.validate_json(path.read_bytes())
