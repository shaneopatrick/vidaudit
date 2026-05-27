"""Tests for vidaudit.description_parser.

spaCy runs for real here — it is deterministic, local, and fast, so there is
nothing to mock (only real VLM/ffmpeg calls are forbidden in tests).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest

from vidaudit.description_parser import (
    Claim,
    DescriptionSegment,
    extract_claims,
    parse_descriptions,
)

if TYPE_CHECKING:
    from pathlib import Path


def _by_type(claims: list[Claim], claim_type: str) -> set[str]:
    return {c.text for c in claims if c.claim_type == claim_type}


def test_extract_claims_basic_objects_and_entity() -> None:
    desc = "Sarah wears a red jacket and holds a coffee cup."
    claims = extract_claims(desc)

    assert {"red jacket", "coffee cup"} <= _by_type(claims, "object")
    assert "Sarah" in _by_type(claims, "entity")
    # "Sarah" is a named entity, so it must not also appear as an object claim.
    assert "sarah" not in _by_type(claims, "object")
    assert all(c.source_description == desc for c in claims)


def test_extract_claims_empty_returns_empty() -> None:
    assert extract_claims("") == []
    assert extract_claims("   \n\t ") == []


def test_extract_claims_filters_generic_phrases() -> None:
    # "camera" and "scene" are medium/composition heads, not visible content.
    assert extract_claims("The camera slowly pans across the empty scene.") == []


def test_extract_claims_dedupes_object_against_entity_with_article() -> None:
    """Smoke regression: 'the Eiffel Tower' (entity) must subsume 'eiffel tower' (object).

    The entity span includes the leading article; the object form strips it.
    Without normalising the entity for comparison, the same landmark gets
    verified twice — wasting a VLM call and double-counting toward
    hallucination_count.
    """
    claims = extract_claims("A woman walks past the Eiffel Tower.")
    eiffel_mentions = [c for c in claims if "eiffel" in c.text.lower()]

    assert len(eiffel_mentions) == 1, (
        f"expected one Eiffel Tower claim after dedup, got: "
        f"{[(c.text, c.claim_type) for c in eiffel_mentions]}"
    )


def test_extract_claims_dedupes_subspan() -> None:
    claims = extract_claims("There is a jacket. There is a red jacket.")
    objects = _by_type(claims, "object")

    assert "red jacket" in objects
    assert "jacket" not in objects  # subsumed by the longer span


def test_parse_descriptions_populates_and_preserves_none(tmp_path: Path) -> None:
    path = tmp_path / "descs.json"
    path.write_text(
        json.dumps(
            [
                {
                    "timestamp_start": 12.5,
                    "timestamp_end": 18.0,
                    "description": "A woman holds a coffee cup",
                },
                {
                    "timestamp_start": 40.0,
                    "timestamp_end": None,
                    "description": "A street musician plays a saxophone",
                },
            ]
        )
    )

    segments = parse_descriptions(path)

    assert [type(s) for s in segments] == [DescriptionSegment, DescriptionSegment]
    assert segments[0].timestamp_end == 18.0
    assert segments[1].timestamp_end is None  # never fabricated
    assert "coffee cup" in _by_type(segments[0].claims, "object")
    assert len(segments[1].claims) > 0


def test_parse_descriptions_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_descriptions(tmp_path / "does_not_exist.json")


def test_parse_descriptions_empty_description_skipped_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "descs.json"
    path.write_text(
        json.dumps([{"timestamp_start": 5.0, "timestamp_end": 9.0, "description": ""}])
    )

    with caplog.at_level(logging.WARNING):
        segments = parse_descriptions(path)

    assert segments[0].claims == []
    assert "empty description" in caplog.text.lower()


def test_models_round_trip() -> None:
    segment = DescriptionSegment(
        timestamp_start=1.0,
        timestamp_end=None,
        description="A dog runs",
        claims=[Claim(text="dog", claim_type="object", source_description="A dog runs")],
    )

    restored = DescriptionSegment.model_validate_json(segment.model_dump_json())

    assert restored == segment
