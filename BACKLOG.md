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
