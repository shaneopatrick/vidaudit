# DESIGN.md — Design Decision Log

A running, append-only log of design decisions for **vidaudit**. Each entry is a
lightweight ADR: what was decided, why, what was rejected, and what it costs.

- **Append-only.** New decisions get the next `DD-N`. Don't rewrite history — if a
  decision changes, add a new entry and mark the old one *Superseded by DD-N*.
- **Relationship to other docs:** `CLAUDE.md` §9 is the terse "don't fix these"
  mirror of the *Accepted* decisions here. `PLAN.md` holds the implementation
  specs. This file holds the *reasoning*. When they disagree, this file is the
  source of truth for *why*; PLAN.md for *how*.

Status values: `Accepted` · `Superseded by DD-N` · `Proposed`.

---

## DD-1: Claims-based verification, not free-text comparison

- **Date:** 2026-05-16 · **Status:** Accepted
- **Decision:** Decompose each description into individual verifiable claims
  (noun phrases, named entities) and verify each independently with a binary VLM
  question ("Is [X] visible in this frame?").
- **Why:** Comparing two generated descriptions compares two noisy text outputs —
  errors compound and the result isn't quantifiable. Per-claim binary checks are
  independent, scorable, and produce a confidence per claim.
- **Rejected:** Generate a second caption and diff/embed-compare it. Kept only as
  the *baseline* the eval measures against (see DD-13).
- **Consequences:** Audit quality is upper-bounded by claim-extraction quality
  (drives DD-2, and the stopword filter in PLAN.md §2).

## DD-2: spaCy for claim extraction, not an LLM

- **Date:** 2026-05-16 · **Status:** Accepted
- **Decision:** Extract claims with spaCy (`en_core_web_sm`) noun-phrase chunking
  + NER, not a second LLM call.
- **Why:** Decomposition should be deterministic, fast, and free. An LLM here
  adds latency, cost, non-determinism, and a second hallucination surface.
- **Consequences:** `en_core_web_sm` emits generic/non-visual phrases ("the
  background", "the center"); a stopword filter is required since extraction
  precision bounds the whole tool (DD-1).

## DD-3: Pluggable VLM backends via ABC; default = Gemini 2.5 Flash

- **Date:** 2026-05-16 · **Status:** Accepted · *Default refined by DD-16 (2026-05-19)*
- **Decision:** `VLMBackend` abstract base class. Default backend is **Gemini
  2.5 Flash** (free tier) via `google-genai`. Qwen2.5-VL is an optional local
  backend, opt-in.
- **Why:** A pluggable interface lets the eval swap models without touching the
  pipeline. Gemini free tier removes the cost barrier for a portfolio project.
- **Note:** "Gemini 2.5" is the single agreed model string across all docs —
  earlier drafts inconsistently said 2.0 / 3.1; that is resolved.

## DD-4: Frame extraction via ffmpeg subprocess (not opencv/decord)

- **Date:** 2026-05-16 · **Status:** Accepted
- **Decision:** Extract frames by shelling out to `ffmpeg`, list-form
  `subprocess.run()`, never shell strings.
- **Why:** Keeps the dependency footprint small and avoids C-extension build
  pain (opencv/decord wheels). ffmpeg is a documented system prerequisite.

## DD-5: All structured data is Pydantic v2 models

- **Date:** 2026-05-16 · **Status:** Accepted
- **Decision:** Every input, output, and cross-component intermediate is a
  Pydantic `BaseModel`. No `dataclasses` for structured data.
- **Why:** Validation at boundaries (untrusted JSON, CLI args) and clean JSON
  serialization for the report come for free.

## DD-6: Batch verification to conserve API quota

- **Date:** 2026-05-16 · **Status:** Accepted
- **Decision:** Multiple claims against the same frame are sent in one VLM
  prompt asking for a JSON array, not one call per claim.
- **Why:** The Gemini free tier is rate-limited; the eval makes many calls.
  Batching is the difference between a feasible and an infeasible eval run.

## DD-7: `confidence` is the VLM's confidence in its verdict

- **Date:** 2026-05-16 · **Status:** Accepted
- **Decision:** `VerificationResult.confidence` ∈ [0,1] is the VLM's confidence
  *in the verdict it gave* (1 = certain, 0 = guess) — **not** P(claim is true).
  The prompt states this explicitly.
- **Why:** Earlier the field was undefined, and `object_audit` branches on it
  (low-confidence "unsupported" → treated as *uncertain*, not flagged).
  Undefined semantics here silently corrupt grounding scores.

## DD-8: Frame-accurate seeking (`-ss` after `-i`)

- **Date:** 2026-05-16 · **Status:** Accepted
- **Decision:** `ffmpeg -i {video} -ss {t} -frames:v 1 ...` — seek *after* input.
- **Why:** `-ss` before `-i` is fast but snaps to the nearest keyframe and can
  return a frame up to a GOP away from `t`. For an auditor, sampling the wrong
  frame manufactures false hallucinations. Accuracy beats speed here.

## DD-9: Context frames cover the segment span, not a ±1s point

- **Date:** 2026-05-16 · **Status:** Accepted *(supersedes the original
  "context frames = t-1s, t+1s" point model)*
- **Decision:** A description covers a time range. Sample the primary frame at
  the segment midpoint and context frames evenly across
  `[timestamp_start, timestamp_end]`. When `timestamp_end` is `None`, resolve an
  effective end in order: (1) next segment's `timestamp_start`; (2) last segment
  → video duration (ffprobe); (3) cap at `max_segment_span` (default 30s). If
  the span collapses to ~0, fall back to point sampling `t ± context_window`.
- **Why:** A claim may be true only briefly within a multi-second segment;
  ±1s around one instant falsely flags it. Inferring the end from the next
  segment's start uses the data's own contiguous structure rather than an
  arbitrary window.
- **Constraints:** Inferred ends MUST be recorded (`end_inferred: true`,
  per-segment) — never silently fabricated. Sampling must be deterministic
  (see DD-14). Resolution runs in the audit orchestration (it needs the ordered
  segment list + video), not in `audit_segment`.

## DD-10: Robust structured output; regex is fallback only

- **Date:** 2026-05-16 · **Status:** Accepted
- **Decision:** Use the SDK's structured output
  (`response_mime_type="application/json"` + `response_schema=<PydanticModel>`)
  so responses are valid JSON *and* typed at the boundary. Pass Pydantic
  `BaseModel` classes directly as `response_schema` — not raw JSON-Schema
  dicts, not prompt-engineered "respond with JSON" hints. Per-field semantic
  guidance (e.g. the precise meaning of `confidence`, DD-7) lives in Pydantic
  `Field(description=...)` so the schema description shipped to the model and
  the data class share one source of truth. Regex extraction is a last-resort
  fallback only, not the primary path.
- **Why:** Prompt-and-pray + regex is fragile and a poor engineering signal
  for a portfolio piece. Native structured output is more robust and
  reproducible. Pydantic schemas additionally give validated, typed Python
  objects back from the SDK (`response.parsed`), removing one layer of manual
  parsing and keeping the model contract co-located with the data class.

## DD-11: Cache VLM verifications, not just frames

- **Date:** 2026-05-16 · **Status:** Accepted
- **Decision:** Cache verification results keyed by (frame content hash, claim
  text), in addition to caching extracted frames.
- **Why:** The eval is iterated repeatedly under a rate-limited free tier.
  Without a verification cache, every rerun re-spends the entire API budget.

## DD-12: Verdict thresholds are eval-derived defaults, not asserted

- **Date:** 2026-05-16 · **Status:** Accepted
- **Decision:** The grounding-score cutoffs (clean / partial / full) and the
  confidence threshold are CLI-tunable *defaults*. The shipped values are chosen
  from an ROC threshold sweep on eval data, not hardcoded by intuition.
- **Why:** For a benchmarking artifact, deriving thresholds from data is the
  rigor the target audience scrutinizes; magic constants undercut credibility.

## DD-13: Eval = baseline comparison + real & synthetic hallucinations + split metrics

- **Date:** 2026-05-16 · **Status:** Accepted
- **Decision:** The eval (a) includes the DD-1-rejected text-comparison approach
  as an explicit baseline and reports vidaudit vs baseline side by side;
  (b) uses *plausible, context-consistent* synthetic mutations **and** a small
  set of real (naturally-generated) hallucinations, reported as separate
  subsets; (c) separates extraction quality from verification quality so a low
  F1 is attributable.
- **Why:** "We catch X% of random swaps" is not credible to a benchmarking
  reviewer. Beating the naive baseline on *realistic* errors is the headline
  result; conflated metrics hide where failures come from.

## DD-14: Reproducibility is a hard requirement

- **Date:** 2026-05-16 · **Status:** Accepted
- **Decision:** Commit `uv.lock` (it is *not* gitignored). Pin an exact VLM
  model ID. Set VLM `temperature=0`. Frame sampling and end-resolution (DD-9)
  are deterministic.
- **Why:** The eval is the deliverable; a deliverable a reviewer can't
  reproduce is worthless. Non-deterministic VLM output or an unpinned
  environment makes reported metrics unfalsifiable.

## DD-15: The eval is the primary deliverable (P0)

- **Date:** 2026-05-16 · **Status:** Accepted
- **Decision:** Treat the FineVideo eval (with DD-13 baseline + DD-12 sweep) as
  P0 — not cuttable. Cut tooling polish, the Colab notebook, and the Qwen
  backend before touching it.
- **Why:** This is a portfolio piece for a benchmarking-focused role. A rough
  tool with a rigorous, baseline-compared eval beats a polished tool with a
  hand-wavy one. The cut list in PLAN.md reflects this ordering.

## DD-16: Canonical backend is open-weight Qwen2.5-VL-3B; Gemini is dev/fallback

- **Date:** 2026-05-19 · **Status:** Accepted *(refines DD-3)*
- **Decision:** Reported eval metrics (DD-15) run on **Qwen2.5-VL-3B-Instruct**
  via `transformers`. Gemini 2.5 Flash is retained as a development backend
  and a no-GPU fallback for users who can't run a local VLM. The Qwen backend
  is no longer a stub — it ships as a real implementation.
- **Why:** The target role hires for "ability to identify the best
  open-weights model for a given task" and explicit VLM expertise; defaulting
  to a closed model contradicts the stated hiring criterion. Open weights are
  also more reproducible (DD-14) — a Qwen checkpoint is frozen by hash forever,
  whereas a Gemini model ID can be deprecated or shift behavior. The
  cross-backend comparison itself becomes the headline eval result (DD-13).
- **Why 3B and not 7B/72B:** 3B fits Colab's free T4 in fp16 (~7 GB) and
  consumer GPUs in 4-bit (~4 GB), so a reviewer can actually reproduce the
  numbers. 7B is run as a scaling-comparison data point if quota permits
  (BACKLOG).
- **Consequences:** Dual-backend dev workflow — Gemini iterated locally on
  Intel/no-GPU machines, Qwen developed and exercised via Colab. The README
  must document both paths. `transformers` + `torch` stay in the `qwen`
  optional extra so the no-GPU install path remains lean.
- **Rejected:** Open-weight-only (drop Gemini). Removes the local-dev
  feedback loop on machines without CUDA/MPS and adds friction for a reviewer
  without GPU access.
- **Not the verifier (clarification re: JD references):** V-JEPA and
  VideoMAE are *representation* models, not VLMs — no language conditioning,
  no VQA capability. They cannot answer "Is X visible?" and so are not
  candidates for the verifier role. A stretch use (temporal-saliency frame
  sampling) is recorded in BACKLOG. Similarly, QwQ-32B is a text-only
  reasoning model, not a VLM — easy to confuse with the Qwen-VL family but
  unrelated.
