# Backlog

Planned work, deferred improvements, and items that are scoped but not yet
scheduled.

---

## description_parser

- **Attribute-claim extraction.** Adjectival modifiers ("red", "wooden") as a
  distinct `claim_type="attribute"` are a stretch goal. Deferred because object
  + entity claims are the core signal, and naive attribute claims
  duplicate/dilute object claims and risk hurting eval precision. Revisit with a
  dedicated attribute-verification prompt.
- **Non-English warning.** A warning on non-English input would be nice, but
  reliable language detection needs an extra dependency (langdetect/fasttext)
  that conflicts with the minimal-deps principle. We currently process anyway
  and spaCy degrades gracefully; add detection only if it earns its dependency.

## VLM backends

- **V-JEPA / VideoMAE for temporal-saliency frame sampling.** These are video
  *representation* models (not VLMs), so they do not replace the verifier — but
  their embeddings could drive smart frame selection: sample frames where the
  embedding changes most across a segment, instead of a fixed `t ± Δ`.
  Better-chosen frames mean less work for the VLM and fewer false flags from
  uninformative samples. Stretch goal.
- **Qwen 7B vs 3B scaling comparison.** Run the eval on both `Qwen2.5-VL-3B` and
  `Qwen2.5-VL-7B` and report the precision/recall delta on hallucination
  detection — a concrete data point on which open-weight model is best for this
  task. 7B fits a Colab free T4 in 4-bit.
- **Additional open-weight VLM comparisons.** InternVL3-2B/8B, PaliGemma 2,
  Pixtral-12B as further data points in the cross-model eval. Only worth doing
  once the 3B baseline is solid.

## Auditor capabilities

- **Action / event verification (multi-frame temporal reasoning).** Current
  single-frame verification is precise on static visual facts (objects,
  entities, attributes) but structurally weak on action/event claims like
  "passing each other", "walking towards", "talking to" — an action exists
  *across time*, not in any one frame.
  - **Worked example** (smoke run, 2026-05-19): a description with
    *"Two yellow tram cars pass each other"* over a 5s segment. The two-tram
    moment is brief (~1.5s) and not present in the segment's primary or
    boundary frames. The VLM correctly returned "a single long yellow train,
    not two distinct tram cars" on every sampled frame and the auditor flagged
    it — *correct given the single-frame contract, but it missed the true brief
    event*.
  - **Possible fixes:**
    1. Denser segment sampling (N=5+ frames across long segments) — a boundary
       fix; doesn't address actions in principle.
    2. Verb-aware claim decomposition: when a verb relates two noun phrases,
       emit static sub-claims plus an action-claim marker.
    3. A separate action-verifier path using a video-native VLM (e.g.
       Qwen2.5-VL with multi-frame input, or VideoLLaMA-style models) for action
       claims only.
  - **For the eval:** report static-claim and action-claim subsets separately so
    the static-claim numbers aren't dragged down by an acknowledged gap.

## Evaluation

- **Scale beyond the pilot.** The current run is 5 videos / 75 synthetic + 30
  real samples (6 real positives) — directional, not statistically tight.
  Larger draws would tighten the real-subset numbers.
- **Broaden synthetic coverage.** The curated object/attribute swap tables
  under-match domain-specific vocabularies (e.g. FineVideo's construction
  footage), so the synthetic set skews toward entity injection. Expand the
  tables or derive swaps from the corpus itself.
