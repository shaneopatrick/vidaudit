# Backlog

Planned work, deferred improvements, and items that are scoped but not yet scheduled.
Items that represent known workarounds or technical shortcuts belong in `TECH_DEBT.md`.
Items here are clean-slate additions or feature completions, not debt paydown.

---

## description_parser

- **Attribute-claim extraction.** PLAN.md §2 marks `claim_type="attribute"`
  (adjectival modifiers like "red", "wooden") a stretch goal. Deferred from
  step 2: object + entity claims are the core signal, and naive attribute
  claims duplicate/dilute object claims and risk hurting eval precision
  (DD-13). Revisit with a dedicated attribute-verification prompt.
- **Non-English warning.** PLAN.md §2 edge case wants a warning on non-English
  input. Reliable language detection needs an extra dependency
  (langdetect/fasttext), which conflicts with the minimal-deps principle
  (DD-4 spirit). Currently we process anyway and spaCy degrades gracefully;
  add detection only if it earns its dependency.

## VLM backends

- **V-JEPA / VideoMAE for temporal-saliency frame sampling.** V-JEPA and
  VideoMAE are video *representation* models (not VLMs), so they do not
  replace the verifier — but their embeddings could drive smart frame
  selection: sample frames where the embedding changes most across a segment,
  instead of fixed `t ± Δ`. Better-chosen frames mean less work for the VLM
  and fewer false flags from uninformative samples. Acknowledges the JD-cited
  model family and is a real ML-eng signal; stretch goal.
- **Qwen 7B vs 3B scaling comparison.** Run the canonical eval on both
  `Qwen2.5-VL-3B` and `Qwen2.5-VL-7B` and report the precision/recall delta
  on hallucination detection. Concrete benchmarking deliverable that directly
  speaks to "ability to identify the best open-weights model for a given
  task" — 7B fits Colab's free T4 in 4-bit.
- **Additional open-weight VLM comparisons.** InternVL3-2B/8B, PaliGemma 2,
  Pixtral-12B as further data points in the cross-model eval. Only worth
  doing once the 3B baseline is solid.

## Auditor capabilities

- **Action / event verification (multi-frame temporal reasoning).** Current
  single-frame verification (DD-1) is precise on static visual facts (objects,
  entities, attributes) but structurally weak on action/event claims like
  "passing each other", "walking towards", "talking to" — an action exists
  *across time*, not in any one frame.
  - **Worked example** (smoke run, 2026-05-19): a description with
    *"Two yellow tram cars pass each other"* over a 5s segment. The two-tram
    moment is brief (~1.5s) and not present in the segment's primary or
    boundary frames. Gemini correctly returned "a single long yellow train,
    not two distinct tram cars" on every sampled frame and the auditor
    flagged it as a hallucination — *correct given the single-frame contract,
    but missed the true brief event*.
  - **Possible fixes (post-MVP):**
    1. Denser segment sampling (N=5+ frames spread across long segments) —
       boundary fix, doesn't address actions principle-ly.
    2. Verb-aware claim decomposition in the parser — when a verb relates two
       noun phrases, emit static sub-claims and an action-claim marker.
    3. A separate action-verifier path using a video-native VLM (e.g.,
       Qwen2.5-VL with multi-frame input, or VideoLLaMA-3) for action claims
       only.
  - **For the eval:** report static-claim and action-claim subsets separately
    so the static-claim numbers aren't dragged down by an acknowledged gap.
