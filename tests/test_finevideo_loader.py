"""Tests for the FineVideo eval loader.

No network, no GPU: ``load_finevideo`` (which streams the real dataset) is
not exercised here — the pure ``chapters_from_row`` extractor is. The
captioner is a stub callable, and frames are tiny in-memory PIL images.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from PIL import Image

from eval.finevideo_loader import (
    EvalSample,
    FineVideoChapter,
    MutationType,
    _mutate_attribute,
    _mutate_entity_injection,
    _mutate_object_swap,
    _replace_first_word,
    build_real_samples,
    chapters_from_row,
    load_dataset,
    make_synthetic_samples,
    save_dataset,
)

if TYPE_CHECKING:
    from pathlib import Path


def _chapter(description: str, *, video_id: str = "vid1") -> FineVideoChapter:
    return FineVideoChapter(
        video_id=video_id,
        timestamp_start=1.0,
        timestamp_end=5.0,
        description=description,
    )


# ---- _replace_first_word --------------------------------------------------


def test_replace_first_word_only_first_occurrence() -> None:
    assert _replace_first_word("a dog and a dog", "dog", "cat") == "a cat and a dog"


def test_replace_first_word_respects_word_boundary() -> None:
    # "dogged" must not match "dog".
    assert _replace_first_word("a dogged dog", "dog", "cat") == "a dogged cat"


# ---- object swap ----------------------------------------------------------


def test_object_swap_replaces_head_noun_with_plausible_alternative() -> None:
    sample = _mutate_object_swap(_chapter("A dog runs across the grass."))

    assert sample is not None
    assert sample.mutation_type is MutationType.object_swap
    assert sample.original_span == "dog"
    assert sample.mutated_span == "cat"
    assert sample.mutated_description == "A cat runs across the grass."
    assert sample.clean_description == "A dog runs across the grass."
    assert sample.source == "synthetic"


def test_object_swap_returns_none_when_no_known_noun() -> None:
    assert _mutate_object_swap(_chapter("The committee discussed policy.")) is None


# ---- attribute mutation ---------------------------------------------------


def test_attribute_mutation_changes_colour_and_labels_full_phrase() -> None:
    sample = _mutate_attribute(_chapter("A red car is parked outside."))

    assert sample is not None
    assert sample.mutation_type is MutationType.attribute_mutation
    assert sample.original_span == "red car"
    assert sample.mutated_span == "blue car"
    assert sample.mutated_description == "A blue car is parked outside."


def test_attribute_mutation_returns_none_without_known_adjective() -> None:
    assert _mutate_attribute(_chapter("A car is parked outside.")) is None


# ---- entity injection -----------------------------------------------------


def test_entity_injection_appends_absent_landmark() -> None:
    sample = _mutate_entity_injection(_chapter("A man walks down the street."))

    assert sample is not None
    assert sample.mutation_type is MutationType.entity_injection
    assert sample.original_span is None
    assert sample.mutated_span == "the Eiffel Tower"
    assert "the Eiffel Tower" in sample.mutated_description
    assert sample.mutated_description.startswith("A man walks down the street.")


def test_entity_injection_skips_already_present_entity() -> None:
    sample = _mutate_entity_injection(_chapter("We toured the Eiffel Tower today."))

    assert sample is not None
    # First candidate is present, so it falls through to the next one.
    assert sample.mutated_span == "the Statue of Liberty"


# ---- make_synthetic_samples -----------------------------------------------


def test_make_synthetic_samples_includes_clean_control_and_mutations() -> None:
    samples = make_synthetic_samples(_chapter("A red dog runs across the grass."))

    # Clean control + object swap + attribute + entity injection = 4.
    assert len(samples) == 4
    clean = samples[0]
    assert clean.mutation_type is None
    assert clean.clean_description == clean.mutated_description

    types = {s.mutation_type for s in samples[1:]}
    assert types == {
        MutationType.object_swap,
        MutationType.attribute_mutation,
        MutationType.entity_injection,
    }
    assert all(s.source == "synthetic" for s in samples)


def test_make_synthetic_samples_can_omit_clean_control() -> None:
    samples = make_synthetic_samples(_chapter("A red dog runs."), include_clean=False)
    assert all(s.mutation_type is not None for s in samples)


def test_make_synthetic_samples_skips_inapplicable_mutations() -> None:
    # No swappable noun, no colour adjective — only entity injection applies.
    samples = make_synthetic_samples(
        _chapter("The committee discussed policy."), include_clean=False
    )
    assert len(samples) == 1
    assert samples[0].mutation_type is MutationType.entity_injection


# ---- build_real_samples ---------------------------------------------------


def _frame() -> Image.Image:
    return Image.new("RGB", (8, 8), color="green")


def test_build_real_samples_pairs_caption_with_ground_truth() -> None:
    chapters = [_chapter("A dog runs in a park.")]

    def captioner(image: Image.Image) -> str:
        return "A large cat sits on a bench."

    samples = build_real_samples(chapters, captioner, frame_for=lambda _c: _frame())

    assert len(samples) == 1
    s = samples[0]
    assert s.source == "real"
    assert s.mutation_type is None
    assert s.clean_description == "A dog runs in a park."
    assert s.mutated_description == "A large cat sits on a bench."


def test_build_real_samples_skips_when_no_frame() -> None:
    chapters = [_chapter("A dog runs.")]
    samples = build_real_samples(chapters, lambda _i: "caption", frame_for=lambda _c: None)
    assert samples == []


def test_build_real_samples_skips_empty_caption() -> None:
    chapters = [_chapter("A dog runs.")]
    samples = build_real_samples(chapters, lambda _i: "   ", frame_for=lambda _c: _frame())
    assert samples == []


# ---- chapters_from_row ----------------------------------------------------


def test_chapters_from_row_extracts_scenes_from_dict() -> None:
    row = {
        "id": "vid1",
        "json": {
            "content_metadata": {
                "scenes": [
                    {"start_time": 1.0, "end_time": 5.0, "description": "A dog runs."},
                    {"start_time": 6.0, "description": "A cat sleeps."},
                ]
            }
        },
    }

    chapters = chapters_from_row(row, "vid1")

    assert len(chapters) == 2
    assert chapters[0].description == "A dog runs."
    assert chapters[0].timestamp_end == 5.0
    assert chapters[1].timestamp_end is None  # missing end tolerated


def test_chapters_from_row_decodes_json_string() -> None:
    row = {
        "json": json.dumps(
            {"content_metadata": {"scenes": [{"start": "2.5", "description": "x"}]}}
        )
    }

    chapters = chapters_from_row(row, "vid2")

    assert len(chapters) == 1
    assert chapters[0].timestamp_start == 2.5  # string coerced to float


def test_chapters_from_row_skips_scenes_missing_description_or_start() -> None:
    row = {
        "json": {
            "content_metadata": {
                "scenes": [
                    {"start_time": 1.0},  # no description
                    {"description": "no start"},  # no start
                    {"start_time": 3.0, "description": "kept"},
                ]
            }
        }
    }

    chapters = chapters_from_row(row, "vid3")

    assert len(chapters) == 1
    assert chapters[0].description == "kept"


def test_chapters_from_row_joins_activities_when_no_description() -> None:
    row = {
        "json": {
            "content_metadata": {
                "scenes": [
                    {
                        "start_time": 1.0,
                        "activities": [
                            {"description": "running"},
                            {"description": "jumping"},
                        ],
                    }
                ]
            }
        }
    }

    chapters = chapters_from_row(row, "vid4")

    assert chapters[0].description == "running jumping"


# ---- dataset I/O ----------------------------------------------------------


def test_dataset_round_trip(tmp_path: Path) -> None:
    samples = [
        EvalSample(
            video_id="vid1",
            timestamp_start=1.0,
            timestamp_end=5.0,
            clean_description="A dog runs.",
            mutated_description="A cat runs.",
            mutation_type=MutationType.object_swap,
            original_span="dog",
            mutated_span="cat",
            source="synthetic",
        ),
        EvalSample(
            video_id="vid1",
            timestamp_start=6.0,
            clean_description="A child plays.",
            mutated_description="A small child plays near a fountain.",
            source="real",
        ),
    ]
    path = tmp_path / "ds" / "samples.json"

    save_dataset(samples, path)
    loaded = load_dataset(path)

    assert loaded == samples
