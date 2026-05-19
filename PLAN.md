# PLAN.md — vidaudit Weekend MVP

## What This Is

A Python CLI tool that audits VLM-generated video descriptions for hallucinations. Given a
video file and a set of time-coded text descriptions (JSON), it samples frames at each
timestamp, decomposes descriptions into verifiable claims, uses a VLM to check each claim
against the actual frame, and produces a structured audit report.

## Why It Exists

Production video indexing systems (like Moments Lab's MXT-2) generate automated time-coded
descriptions of video content. These descriptions sometimes hallucinate — mentioning objects,
people, or landmarks that aren't actually in the frame. There is no lightweight, pip-installable
tool to audit these descriptions against ground truth frames. Academic work in this space
(VideoHallucer, ViBe, MESH) produces benchmarks and datasets, not reusable evaluation libraries.

vidaudit fills that gap.

## Core Design Insight

Instead of generating a second description and comparing two noisy text outputs, vidaudit
decomposes the original description into individual claims (noun phrases, named entities) and
asks the VLM binary verification questions: "Is [X] visible in this frame?" This reduces the
audit to a set of independent, quantifiable checks with confidence scores.

---

## Components

### 1. Frame Sampler (`vidaudit/frame_sampler.py`)

**Purpose:** Extract frames from a video at specific timestamps.

**Interface:**
```python
def sample_frames(
    video_path: Path,
    timestamps: list[float],
    context_window: float = 1.0,
) -> dict[float, list[Image.Image]]:
    """Extract frames at each timestamp, plus neighboring frames within context_window.

    Args:
        video_path: Path to video file (mp4, mov, mkv, etc.)
        timestamps: List of timestamps in seconds
        context_window: Seconds before/after to sample additional context frames

    Returns:
        Dict mapping each timestamp to a list of PIL Images.
        Primary frame is index 0, context frames follow.
    """
```

**Implementation:**
- Use ffmpeg via subprocess. Put `-ss` AFTER `-i` for frame-accurate seeking:
  `ffmpeg -i {video} -ss {timestamp} -frames:v 1 -f image2pipe -`. `-ss` before
  `-i` is fast but snaps to the nearest keyframe and can return a frame up to a
  GOP away from the requested time — unacceptable for an auditor.
- For each timestamp, extract 3 frames evenly spaced within context_window:
  t-Δ, t, t+Δ
- Return PIL Images (read from ffmpeg stdout pipe)
- Cache extracted frames to `VIDAUDIT_CACHE_DIR` to avoid re-extraction on reruns
- Validate video exists and is readable before processing

**Dependencies:** ffmpeg (system), Pillow

**Edge cases:**
- Timestamp beyond video duration → skip with warning
- Corrupted video → raise clear error with video path
- Very short videos where t-1s < 0 → clamp to 0

---

### 2. Description Parser (`vidaudit/description_parser.py`)

**Purpose:** Parse time-coded descriptions from JSON and extract verifiable claims.

**Input format:**
```json
[
  {
    "timestamp_start": 12.5,
    "timestamp_end": 18.0,
    "description": "A woman in a red jacket walks past the Eiffel Tower while holding a coffee cup"
  }
]
```

**Output models:**
```python
class Claim(BaseModel):
    text: str                          # "red jacket", "Eiffel Tower", "coffee cup"
    claim_type: Literal["object", "entity", "attribute"]
    source_description: str            # full original description

class DescriptionSegment(BaseModel):
    timestamp_start: float
    timestamp_end: float | None
    description: str
    claims: list[Claim]
```

**Implementation:**
- Load and validate JSON input with Pydantic
- Run spaCy `en_core_web_sm` on each description
- Extract noun phrases → `claim_type="object"`
- Extract named entities (PERSON, ORG, GPE, FAC, LOC) → `claim_type="entity"`
- Extract adjectival modifiers on nouns → `claim_type="attribute"` (stretch goal)
- Deduplicate overlapping spans (e.g., "red jacket" and "jacket")
- Filter non-visual / generic phrases via a stopword list ("the frame", "the
  background", "the scene", "the camera", "the center", pronouns, bare
  determiners). `en_core_web_sm` chunking emits these and they poison precision
  — extraction quality upper-bounds the whole tool's accuracy.

**Dependencies:** spaCy, en_core_web_sm model

**Edge cases:**
- Empty description → skip with warning
- No extractable claims → return segment with empty claims list
- Non-English text → warn and attempt anyway (spaCy will degrade gracefully)

---

### 3. VLM Backend Interface (`vidaudit/vlm/base.py`)

**Purpose:** Abstract interface for VLM-based claim verification.

**Interface:**
```python
class VerificationResult(BaseModel):
    claim: str
    verdict: Literal["supported", "unsupported", "uncertain"]
    confidence: float                  # VLM's confidence IN ITS VERDICT (0=guess, 1=certain)
    evidence: str                      # VLM's explanation

class VLMBackend(ABC):
    @abstractmethod
    def verify_claim(self, image: Image.Image, claim: str) -> VerificationResult:
        """Check if a claim is visually supported by the image."""
        pass

    @abstractmethod
    def verify_batch(self, image: Image.Image, claims: list[str]) -> list[VerificationResult]:
        """Verify multiple claims against the same image. Default: sequential calls."""
        pass
```

**Design notes:**
- `verify_batch` exists for backends that support batched prompts (saves API calls)
- Default `verify_batch` implementation loops over `verify_claim`
- All backends must return structured `VerificationResult` — never raw text

---

### 4. Gemini Backend (`vidaudit/vlm/gemini.py`)

**Purpose:** Default VLM backend using Gemini 2.5 Flash (free tier).

**Implementation:**
- Use `google-genai` SDK
- Send image + structured prompt per claim
- Prompt template:

```
You are a video frame auditor. You will be shown a single frame from a video
and a claim about what appears in that frame.

Determine whether the claim is visually supported by the frame.

Claim: "{claim}"
```

Per DD-10, the response shape is enforced by `response_schema=<PydanticModel>`,
not by the prompt. Per-field semantic guidance (e.g. the definition of
`confidence`, DD-7) lives in `Field(description=...)` on the response model and
is shipped to Gemini via the schema — keep it out of the prompt.

- Use the SDK's structured output (`response_mime_type="application/json"` +
  `response_schema`) so responses are valid JSON by construction; regex
  extraction is a last-resort fallback only, not the primary path
- Set `temperature=0` and pin an exact model ID — eval runs must be reproducible
- Cache verification results keyed by (frame content hash, claim text) so reruns
  and eval iteration don't re-spend the API quota
- Rate limiting: respect the free-tier req/min limit, add sleep between calls
- Batch optimization: for multiple claims on same frame, combine into single prompt
  asking for a JSON array of verdicts (saves API calls significantly)

**Combined batch prompt:**
```
You are a video frame auditor. You will be shown a single frame from a video
and a list of claims about what appears in that frame.

For EACH claim, determine whether it is visually supported by the frame.

Claims:
1. "{claim_1}"
2. "{claim_2}"
3. "{claim_3}"

Respond with ONLY a JSON array, no other text:
[
  {{"claim": "{claim_1}", "verdict": "supported"|"unsupported"|"uncertain", "confidence": 0.0-1.0, "evidence": "..."}},
  ...
]
```

**Dependencies:** google-genai

**Edge cases:**
- API key missing → clear error: "Set GEMINI_API_KEY environment variable"
- Rate limit hit → exponential backoff with max 3 retries
- Malformed JSON response → retry once, then return verdict="uncertain" with confidence=0.0

---

### 5. Qwen VL Backend (`vidaudit/vlm/qwen_vl.py`)

**Purpose:** **Primary** backend for reported eval metrics (DD-16). Open-weight
Qwen2.5-VL-3B-Instruct via `transformers`. Reproducibility-anchored — a frozen
checkpoint hash, no risk of silent provider-side behavior change.

**Implementation:**
- Use `Qwen/Qwen2.5-VL-3B-Instruct`. 3B is chosen so the canonical eval
  reproduces on a Colab free T4 (fp16 ~7 GB) or a consumer GPU in 4-bit
  (~4 GB) — see DD-16.
- Same prompt strategy as the Gemini backend (single-claim + batched).
  Greedy decoding (`do_sample=False`) and a pinned model revision (commit
  SHA, not floating `main`) for determinism (DD-14).
- Share the verification cache (DD-11) keyed by (frame hash, claim) with
  the Gemini backend, so backend swaps don't invalidate cached verdicts.
- Optional 7B variant via `VIDAUDIT_QWEN_MODEL` for a scaling-comparison
  data point in the eval (see BACKLOG).

**Dependencies:** `transformers`, `torch`, `accelerate` — kept in the `[qwen]`
optional extra so no-GPU users aren't forced to install ~1.5 GB of CUDA/torch
wheels.

**Dev workflow:** developed and exercised in Colab. The primary authoring
machine is Intel macOS (no CUDA/MPS), so local execution is impractical —
local iteration uses the Gemini backend, canonical eval numbers come from the
Colab notebook running this backend.

**Edge cases:**
- Model not downloaded → clear error with the `huggingface-cli download`
  command
- OOM → suggest 4-bit quantization (`bitsandbytes`) or the 3B variant if on 7B

---

### 6. Object Auditor (`vidaudit/auditors/object_audit.py`)

**Purpose:** Core audit logic — verify object/entity claims against frames.

**Interface:**
```python
class ClaimResult(BaseModel):
    claim: Claim
    verification: VerificationResult
    flagged: bool                      # True if hallucination candidate

class SegmentAuditResult(BaseModel):
    segment: DescriptionSegment
    claim_results: list[ClaimResult]
    grounding_score: float             # verified_claims / total_claims
    hallucination_count: int
    verdict: Literal["clean", "partial_hallucination", "full_hallucination"]

def audit_segment(
    segment: DescriptionSegment,
    frames: list[Image.Image],
    vlm: VLMBackend,
    confidence_threshold: float = 0.3,
) -> SegmentAuditResult:
    """Audit a single description segment against its frames."""
```

**Implementation:**
- Sample the primary frame at the segment midpoint and context frames evenly
  spaced across [timestamp_start, timestamp_end] — claims describe the whole
  span, not one instant
- When `timestamp_end` is None, the sampling step resolves an effective end in
  this order: (1) the next segment's `timestamp_start`; (2) for the last
  segment, the video duration (ffprobe); (3) cap the span at `max_segment_span`
  (default 30s) so a lone timestamp with a large trailing gap doesn't span
  forever. If the resolved span collapses to ~0, fall back to point sampling
  `t ± context_window`. Record `end_inferred: true` in report metadata for any
  segment whose end was inferred — never silently fabricate a span. (This
  resolution needs the ordered segment list + video, so it runs in the audit
  orchestration, not in `audit_segment`.)
- For each claim in the segment, call `vlm.verify_claim` (or `verify_batch`) on the primary frame
- If verdict is "unsupported" with high verdict-confidence (> threshold), check
  the context frames before flagging. A low-confidence "unsupported" is treated
  as "uncertain", not a hallucination.
- If still unsupported across all frames → flag as hallucination
- Compute grounding_score = supported_claims / total_claims
- Verdict thresholds below are DEFAULTS, surfaced as CLI flags. The values
  shipped must be chosen from the eval threshold sweep (§9), not asserted:
  - grounding_score >= 0.8 → "clean"
  - grounding_score >= 0.4 → "partial_hallucination"
  - grounding_score < 0.4 → "full_hallucination"

**Key logic:** checking context frames before flagging prevents false positives from
motion blur or brief occlusion at the exact timestamp.

---

### 7. Report Generator (`vidaudit/report.py`)

**Purpose:** Aggregate audit results and output structured report.

**Output format (JSON):**
```json
{
  "metadata": {
    "video": "sample.mp4",
    "backend": "gemini-2.5-flash",
    "timestamp": "2025-01-15T10:30:00Z",
    "confidence_threshold": 0.3,
    "vidaudit_version": "0.1.0"
  },
  "summary": {
    "total_descriptions": 12,
    "total_claims": 47,
    "verified_claims": 38,
    "hallucinated_claims": 6,
    "uncertain_claims": 3,
    "overall_grounding_score": 0.81,
    "descriptions_flagged": 3
  },
  "segments": [
    {
      "timestamp_start": 12.5,
      "timestamp_end": 18.0,
      "description": "A woman in a red jacket walks past the Eiffel Tower",
      "grounding_score": 0.67,
      "verdict": "partial_hallucination",
      "claims": [
        {
          "text": "woman",
          "type": "object",
          "verdict": "supported",
          "confidence": 0.95,
          "evidence": "A woman is clearly visible in the center of the frame"
        },
        {
          "text": "red jacket",
          "type": "object",
          "verdict": "supported",
          "confidence": 0.88,
          "evidence": "The woman is wearing a red outer garment"
        },
        {
          "text": "Eiffel Tower",
          "type": "entity",
          "verdict": "unsupported",
          "confidence": 0.12,
          "evidence": "The background shows a city street with no recognizable landmarks"
        }
      ]
    }
  ]
}
```

**Terminal output (via Rich):**
- Summary table with overall stats
- Per-segment rows: timestamp | description (truncated) | score | verdict (color-coded)
- Flagged claims highlighted in red
- Clean claims in green

---

### 8. CLI (`vidaudit/cli.py`)

**Purpose:** Typer-based command-line interface.

**Commands:**
```bash
# Primary command
vidaudit audit \
  --video input.mp4 \
  --descriptions descs.json \
  --output report.json \
  --backend gemini \
  --confidence-threshold 0.3 \
  --verbose

# Convenience: just extract and show claims (no VLM, useful for debugging)
vidaudit parse \
  --descriptions descs.json
```

**Implementation:**
- `audit` command orchestrates the full pipeline:
  1. Parse descriptions → extract claims
  2. Sample frames at each timestamp
  3. Run object auditor on each segment
  4. Generate and save report
  5. Print terminal summary
- Show Rich progress bar during VLM verification (the slow step)
- `parse` command runs only the description parser and prints extracted claims
  (useful for verifying NLP extraction before burning API calls)

---

### 9. FineVideo Evaluation (`eval/`)

**Purpose:** Validate vidaudit on real video data with reproducible metrics.

**`eval/finevideo_loader.py`:**
- Load 5-10 videos from FineVideo test split (HuggingFace `datasets` library)
- Extract their ground-truth chapter descriptions
- Generate synthetic hallucinated versions:
  - Object swap: replace a noun phrase with a *plausible, context-consistent*
    object (a likely co-occurring one), not a random object — random swaps are
    trivially detectable and inflate the metrics
  - Entity injection: add a named entity that isn't in the video
  - Attribute mutation: change colors/sizes ("red" → "blue")
- Also collect a small REAL-hallucination set: run a weak captioner over the
  frames and keep its naturally-hallucinated descriptions. Synthetic mutations
  alone don't resemble real VLM error distributions; the real set is what gives
  the eval face validity for this domain.
- Output: pairs of (clean_description, mutated_description) with mutation labels
  and a `source` field ("synthetic" | "real")

**`eval/run_eval.py`:**
- Run vidaudit on both clean and mutated descriptions
- Baseline for comparison: re-caption each frame with the VLM and compare the
  two texts by similarity, flagging low-similarity segments. This is the
  "two noisy descriptions" approach vidaudit argues against — showing
  claims-decomposition beats it is the headline result for this project.
- Report the synthetic and real subsets separately (don't average them away)
- Separate the two failure modes: report spaCy extraction quality on its own,
  and a hand-labeled confusion matrix on the VLM verifier, so a low F1 is
  attributable to extraction vs verification
- Compute metrics:
  - True Positive: mutated claim correctly flagged
  - False Positive: clean claim incorrectly flagged
  - False Negative: mutated claim missed
  - Precision, Recall, F1 — vidaudit vs baseline, side by side
- Threshold sweep: ROC over confidence/grounding thresholds; the defaults
  shipped in object_audit are picked here, not asserted
- Print results table
- Save detailed results to `eval/results/`

**Target:** run this in Google Colab. Include a notebook at `notebooks/eval_demo.ipynb`
that does the full eval and displays results inline.

---

## Build Order

This is the sequence to implement. Each step should be a working commit.

| Step | Component | Est. Time | Depends On |
|------|-----------|-----------|------------|
| 1 | `pyproject.toml` + `Makefile` + `.env.example` + skeleton | 30 min | nothing |
| 2 | `description_parser.py` + `tests/test_description_parser.py` | 1.5 hrs | step 1 |
| 3 | `frame_sampler.py` + `tests/test_frame_sampler.py` | 1.5 hrs | step 1 |
| 4 | `vlm/base.py` + `vlm/gemini.py` (dev/fallback backend) | 2 hrs | step 1 |
| 5 | `auditors/object_audit.py` + `tests/test_object_audit.py` | 2 hrs | steps 2, 3, 4 |
| 6 | `report.py` | 1.5 hrs | step 5 |
| 7 | `cli.py` | 1.5 hrs | steps 5, 6 |
| 8 | End-to-end test on a real video (Gemini backend, local) | 1 hr | step 7 |
| 9 | `vlm/qwen_vl.py` + Colab smoke test (canonical backend, DD-16) | 3 hrs | step 4 |
| 10 | `eval/finevideo_loader.py` + plausible synthetic + real mutations | 2 hrs | step 2 |
| 11 | `eval/run_eval.py` — Colab, compares Qwen vs Gemini vs text-baseline | 2.5 hrs | steps 7, 9, 10 |
| 12 | `README.md` with results table + Colab notebook (reproduction artifact) | 2 hrs | step 11 |
| 13 | Polish: docstrings, demo CLI output snippet, anything missed | 1.5 hr | step 12 |

**Total: ~24 hrs.** If running behind, cut from the "What to Cut" list
bottom-up — never cut the Qwen backend or the cross-backend eval; those are
the deliverable (DD-15, DD-16).

## What to Cut (Priority Order)

This is a portfolio piece for a benchmarking-focused role — the eval IS the
deliverable. A rough tool with a rigorous, baseline-compared eval beats a
polished tool with a hand-wavy one. Cut from the bottom up:

1. **Always keep:** description_parser + frame_sampler + Gemini backend +
   Qwen backend + object_audit + report + FineVideo eval comparing the two
   backends against the text-comparison baseline, with threshold sweep
   (this is the credibility — it is NOT optional)
2. **Keep if possible:** real-hallucination subset; Colab notebook as the
   canonical eval reproduction artifact; CLI `audit` command polish
3. **Demote:** `parse` CLI subcommand — nice to have, not essential
4. **Cut:** Demo GIF — a code block in the README showing CLI output is fine

## Post-Weekend Stretch Goals (Do NOT build this weekend)

- Quantization benchmarking (fp16 vs 4-bit on Qwen backend)
- Action/verb verification (requires multi-frame temporal reasoning)
- Temporal ordering checks (does event A happen before event B as described?)
- FastAPI wrapper for CI/CD integration
- GitHub Actions workflow that runs eval on PR