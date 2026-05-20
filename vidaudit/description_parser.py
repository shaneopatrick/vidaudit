"""Parse time-coded descriptions and decompose them into verifiable claims.

This is the deterministic front of the pipeline (DESIGN.md DD-1, DD-2): each
description is split into independent noun-phrase / named-entity claims using
spaCy, not an LLM. Extraction precision upper-bounds the whole tool's accuracy,
so generic non-visual phrases are filtered and overlapping spans deduped.

`timestamp_end` is preserved exactly as given (including ``None``); resolving a
missing end into an effective span is the orchestration's job, not the
parser's (DD-9).
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Literal

import spacy
from pydantic import BaseModel, Field, TypeAdapter

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_SPACY_MODEL = "en_core_web_sm"

# Named-entity labels that denote something visually checkable in a frame.
_ENTITY_LABELS = frozenset({"PERSON", "ORG", "GPE", "FAC", "LOC"})

# Head nouns that describe the medium/composition rather than visible content.
# spaCy noun-chunk extraction emits these constantly ("the background", "the
# center of the frame") and they poison precision if left in.
_GENERIC_HEADS = frozenset(
    {
        "frame",
        "background",
        "foreground",
        "scene",
        "camera",
        "center",
        "image",
        "video",
        "picture",
        "photo",
        "shot",
        "view",
        "screen",
        "footage",
        "clip",
        "thing",
    }
)

ClaimType = Literal["object", "entity", "attribute"]


class Claim(BaseModel):
    """A single verifiable assertion extracted from a description."""

    text: str
    claim_type: ClaimType
    source_description: str


class DescriptionSegment(BaseModel):
    """One time-coded description plus the claims decomposed from it.

    ``timestamp_end`` is ``None`` when the input omitted it; it is never
    fabricated here (see DESIGN.md DD-9).
    """

    timestamp_start: float
    timestamp_end: float | None = None
    description: str
    claims: list[Claim] = Field(default_factory=list)


_SEGMENTS_ADAPTER = TypeAdapter(list[DescriptionSegment])


@functools.lru_cache(maxsize=1)
def _load_spacy() -> spacy.Language:
    """Load and cache the spaCy pipeline.

    Raises:
        OSError: If the spaCy model is not installed (run ``make install``).
    """
    return spacy.load(_SPACY_MODEL)


def _normalize_span(span: spacy.tokens.Span) -> str:
    """Strip leading determiners/pronouns/possessives and normalise case.

    "A woman" -> "woman", "the Eiffel Tower" -> "eiffel tower". Used both to
    produce object-claim text from noun chunks and to normalise entity spans
    for object-vs-entity dedup (so "the Eiffel Tower" entity subsumes the
    "eiffel tower" object form). Returns "" if nothing visually meaningful
    remains (e.g. a bare pronoun).
    """
    tokens = list(span)
    while tokens and (tokens[0].pos_ in {"DET", "PRON"} or tokens[0].dep_ == "poss"):
        tokens = tokens[1:]
    return " ".join(t.text for t in tokens).strip().lower()


def extract_claims(description: str) -> list[Claim]:
    """Decompose a description into deduplicated, filtered claims.

    Named entities (people, orgs, places, landmarks) become ``entity`` claims;
    remaining noun phrases become ``object`` claims. Generic non-visual phrases
    are dropped and shorter spans subsumed by a longer one are removed
    ("jacket" is dropped in favour of "red jacket").

    Args:
        description: The full natural-language description text.

    Returns:
        Claims in order of first appearance. Empty if nothing extractable.
    """
    if not description.strip():
        return []

    doc = _load_spacy()(description)

    # (start_char, claim_type, text) — start_char gives a deterministic order.
    candidates: list[tuple[int, ClaimType, str]] = []
    # Article-stripped, lowercased entity forms — used to dedup object claims
    # against entities (e.g. drop the "eiffel tower" object when "the Eiffel
    # Tower" is already an entity, even though the strings differ literally).
    entity_normalized: set[str] = set()
    for ent in doc.ents:
        if ent.label_ in _ENTITY_LABELS:
            text = ent.text.strip()
            if text:
                candidates.append((ent.start_char, "entity", text))
                entity_normalized.add(_normalize_span(ent))
    for chunk in doc.noun_chunks:
        if chunk.root.lemma_.lower() in _GENERIC_HEADS:
            continue
        text = _normalize_span(chunk)
        if text and text not in _GENERIC_HEADS:
            candidates.append((chunk.start_char, "object", text))

    # Drop object spans subsumed by a longer object span (whole-word), and any
    # object that is really a named entity. Process longest-first so the most
    # specific phrase wins.
    kept_objects: list[str] = []
    keep: set[tuple[int, ClaimType, str]] = set()
    for cand in sorted(candidates, key=lambda c: len(c[2]), reverse=True):
        _, ctype, text = cand
        if ctype != "object":
            keep.add(cand)
            continue
        low = text.lower()
        if low in entity_normalized:
            continue
        if any(low == k or _is_subspan(low, k) for k in kept_objects):
            continue
        kept_objects.append(low)
        keep.add(cand)

    claims: list[Claim] = []
    seen: set[tuple[ClaimType, str]] = set()
    for start, ctype, text in sorted(candidates, key=lambda c: c[0]):
        if (start, ctype, text) not in keep:
            continue
        dedup_key = (ctype, text.lower())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        claims.append(Claim(text=text, claim_type=ctype, source_description=description))
    return claims


def _is_subspan(needle: str, haystack: str) -> bool:
    """True if ``needle`` appears as a whole-word run inside ``haystack``."""
    if needle == haystack:
        return False
    h_words = haystack.split()
    n_words = needle.split()
    return any(
        h_words[i : i + len(n_words)] == n_words for i in range(len(h_words) - len(n_words) + 1)
    )


def parse_descriptions(path: Path) -> list[DescriptionSegment]:
    """Load, validate, and decompose a descriptions JSON file.

    The input is a JSON array of objects with ``timestamp_start``,
    ``description``, and an optional ``timestamp_end``. Validated with Pydantic
    at the boundary (CLAUDE.md §7).

    Args:
        path: Path to the descriptions JSON file.

    Returns:
        Segments with their ``claims`` populated. Segments with an empty
        description are kept with no claims and a warning is logged.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        pydantic.ValidationError: If the JSON does not match the schema.
    """
    if not path.exists():
        raise FileNotFoundError(f"Descriptions file not found: {path}")

    segments = _SEGMENTS_ADAPTER.validate_json(path.read_bytes())
    for segment in segments:
        if not segment.description.strip():
            logger.warning("Skipping empty description at t=%.2fs", segment.timestamp_start)
            continue
        segment.claims = extract_claims(segment.description)
    return segments
