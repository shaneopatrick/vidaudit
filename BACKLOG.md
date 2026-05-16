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
