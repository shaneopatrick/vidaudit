# vidaudit

**Audit VLM-generated video descriptions for hallucinations.**

`vidaudit` is a CLI tool that checks time-coded video descriptions against the
actual frames. Given a video and a set of timestamped descriptions (JSON), it
samples frames at each timestamp, decomposes each description into individual
verifiable **claims** (objects, named entities, attributes), and asks a
vision-language model a binary question per claim — *"Is this visible in this
frame?"* — producing a structured audit report with a grounding score per
segment.

## Why

Production video-indexing systems generate automated time-coded 
descriptions of video content. Those descriptions sometimes
hallucinate — naming objects, people, or landmarks that aren't actually on
screen. Academic work in this space (VideoHallucer, ViBe, MESH) ships
*benchmarks*, not a reusable evaluation tool. `vidaudit` fills that gap with a
small, inspectable CLI you clone and run.

## Core idea: claims, not text comparison

The naive approach generates a second caption and diffs the two texts — but that
compares two noisy outputs, and errors compound. `vidaudit` instead decomposes
the description into independent claims and verifies each one against the frame
with a binary VLM question. Each check is independent, quantifiable, and carries
its own confidence — that's the project's central design decision. The full set
of design decisions and their tradeoffs is in [DESIGN.md](DESIGN.md).

## Example

```
$ vidaudit audit --video clip.mp4 --descriptions descs.json --backend gemini

──────────────────────────── vidaudit report ────────────────────────────
video:    clip.mp4
backend:  gemini-2.5-flash

╭─ Segment 1.0–4.0s  verdict=clean  grounding=1.00  flagged=0 ──────────────╮
│ A person rides a bicycle towards the camera at a busy city intersection   │
╰───────────────────────────────────────────────────────────────────────────╯
 ✓ person                  object  supported  1.00  ·  Individuals visible…
 ✓ bicycle                 object  supported  1.00  ·  A person riding a bike…
 ✓ busy city intersection  object  supported  0.90  ·  Pedestrians, traffic…

╭─ Segment 4.0–9.0s  verdict=partial_hallucination  grounding=0.67  flagged=1 ╮
│ Two yellow tram cars pass each other as pedestrians cross the street        │
╰───────────────────────────────────────────────────────────────────────────╯
 ✗ two yellow tram cars  object  unsupported  1.00  FLAG  Single long yellow
                                                          train, not two cars.
 ✓ pedestrians           object  supported    1.00  ·     People walking…
 ✓ street                object  supported    1.00  ·     Asphalt road visible…

╭─ Segment 13.0–15.0s  verdict=full_hallucination  grounding=0.25  flagged=3 ╮
│ A person in a red jacket walks past the Eiffel Tower holding a coffee cup   │
╰───────────────────────────────────────────────────────────────────────────╯
 ✓ person           object  supported    1.00  ·     A man walking in frame…
 ✗ red jacket       object  unsupported  0.90  FLAG  No red jacket visible.
 ✗ the Eiffel Tower entity  unsupported  1.00  FLAG  Tower is the Berlin TV
                                                     Tower, not the Eiffel.
 ✗ coffee cup       object  unsupported  0.90  FLAG  No coffee cup in frame.
```

The auditor catches the fabricated Eiffel Tower (it's the Berlin TV Tower), the
nonexistent red jacket and coffee cup, and the miscount of tram cars — while
correctly leaving the grounded claims alone.

## Install

Requires **Python 3.10+**, **ffmpeg** (system install), and
[`uv`](https://docs.astral.sh/uv/).

```bash
# system prerequisite
brew install ffmpeg          # macOS  (apt install ffmpeg on Debian/Ubuntu)

# project
git clone https://github.com/shaneopatrick/vidaudit.git
cd vidaudit
make install                 # uv sync + downloads the spaCy en_core_web_sm model
```

Set up credentials for the default Gemini backend:

```bash
cp .env.example .env         # then add your GEMINI_API_KEY
```

## Usage

```bash
# Full audit → JSON report + terminal summary
vidaudit audit \
  --video clip.mp4 \
  --descriptions descs.json \
  --output report.json

# Just show extracted claims (no VLM calls) — handy for debugging extraction
vidaudit parse --descriptions descs.json
```

Descriptions are a JSON array of timestamped segments (`timestamp_end` is
optional — a missing end is inferred from the next segment's start, or the
video duration for the final segment):

```json
[
  {
    "timestamp_start": 12.5,
    "timestamp_end": 18.0,
    "description": "A woman in a red jacket walks past the Eiffel Tower"
  }
]
```

Key flags: `--backend {gemini,qwen}`, `--confidence-threshold`,
`--clean-threshold`, `--partial-threshold`, `--max-segment-span`,
`--qwen-revision`, `--qwen-4bit`. Run `vidaudit audit --help` for the full list.

## Backends

| Backend | Model | Role |
|---|---|---|
| **Qwen2.5-VL** | `Qwen/Qwen2.5-VL-3B-Instruct` (open-weight) | **Canonical** — reported eval metrics run here |
| **Gemini** | `gemini-2.5-flash` | Dev convenience + no-GPU fallback |

The canonical backend is the open-weight Qwen model: it's reproducible (a
checkpoint is frozen by hash forever, where a hosted model ID can shift
behavior) and the cross-model comparison is itself an eval result. Gemini is
retained for fast local iteration on machines without a GPU.

The Qwen backend is GPU-bound; run it in Colab (the dev machine here is a
no-GPU Intel Mac). See [`notebooks/qwen_smoke.ipynb`](notebooks/qwen_smoke.ipynb)
for a one-clip smoke test.

## How it works

```
video.mp4 + descriptions.json
   │
   ├─ description_parser ─► spaCy noun-phrase + NER claim extraction
   ├─ frame_sampler ──────► ffmpeg frame-accurate extraction, span-aware sampling
   ├─ auditors/object_audit ─► per-claim binary VLM verification, context-frame rescue
   │     └─ vlm/{qwen_vl,gemini} ─► batched, structured output, cached verdicts
   └─ report ─────────────► JSON report + Rich terminal summary
```

## Evaluation

The eval *is* the deliverable for this project (it's a portfolio piece for a
benchmarking-focused role), so it's built to be rigorous and reproducible. It's
framed as a **cross-model
benchmark**: every sample is audited by two verifiers — open-weight
**Qwen2.5-VL-3B** and **Gemini 2.5 Flash** — against a text-comparison baseline,
which lets it answer two questions at once: *which VLM is the better
hallucination auditor*, and *does a model auditing its own output have a blind
spot?*

Methodology choices that matter:

- **Baseline comparison.** Claims-decomposition is measured against the naive
  "re-caption and diff the two texts" approach it argues against.
- **Plausible synthetic mutations**, not random ones — object swaps to a likely
  co-occurring object (`dog`→`cat`), colour/size changes, named-entity
  injection. Random swaps are trivially detectable and inflate the metrics; the
  swap tables are hand-curated and auditable.
- **A real-hallucination subset** harvested by captioning frames with one model
  and keeping its natural errors, then hand-labeling against the frame.
- **Self vs cross audit.** The real captions are generated by Qwen, so the Qwen
  verifier is a *self-audit* and Gemini a *cross-audit* — the gap measures the
  self-consistency blind spot.
- **Subsets reported separately**, never averaged; **extraction quality
  reported on its own** so a low F1 is attributable to spaCy extraction vs VLM
  verification; **thresholds derived from a sweep**, not asserted.

### Results

FineVideo pilot — 5 videos, 75 synthetic samples (41 mutations / 34 clean) and
30 hand-labeled real samples (6 hallucinated / 24 clean). Reproduce with
[`notebooks/eval_demo.ipynb`](notebooks/eval_demo.ipynb). Extraction recall
**0.95** (the recall ceiling below is verification, not extraction).

**Synthetic subset — which verifier audits best:**

| Auditor | Precision | Recall | F1 |
|---|---|---|---|
| **vidaudit · Qwen2.5-VL-3B** (open) | **0.97** | 0.71 | **0.82** |
| vidaudit · Gemini 2.5 Flash | 0.63 | 0.88 | 0.73 |
| text-comparison baseline | 0.55 | 1.00 | 0.71 |

The open-weight 3B model is the **most precise** auditor and wins on F1; Gemini
trades precision for recall; the text baseline flags essentially everything
(recall 1.0, precision 0.55). Claim-decomposition beats text comparison
decisively on precision.

**Real subset — self-audit vs cross-audit** (captions generated by Qwen):

| Auditor | Role | Caught | Precision | Recall | F1 |
|---|---|---|---|---|---|
| vidaudit · Qwen (generator) | **self-audit** | 0 / 6 | 0.00 | 0.00 | 0.00 |
| vidaudit · Gemini | **cross-audit** | 3 / 6 | 0.27 | 0.50 | 0.35 |
| text-comparison baseline | — | 6 / 6 | 0.20 | 1.00 | 0.33 |

**A model can't catch its own hallucinations.** Qwen, asked to verify claims
extracted from its own captions, confirms all of them (0/6). An independent
verifier recovers half (3/6) — though even cross-model auditing is hard here
(precision 0.27). The baseline "catches" everything only by flagging
everything.

**Confidence calibration.** The threshold sweep shows Qwen's confidence is
discriminative (F1 peaks at threshold ≈0.2–0.4 then falls — the shipped default
of **0.3** sits on that plateau), while Gemini's is nearly flat across the
sweep — it reports high confidence on almost every verdict, so its confidence
isn't usable for gating.

**Caveats.** This is a pilot: only 6 real positives, so the real-subset numbers
are directional, not significant. The synthetic mix is dominated by
entity-injection (the curated swap tables under-matched FineVideo's
domain vocabulary), and cross-model auditing helps but doesn't solve the real
case. These are honest limits, not hidden ones — see the notebook for the full
confusion matrices and per-verifier sweeps.

Locally (no GPU) you can run the eval with the Gemini backend only:

```bash
make eval     # uv run python eval/run_eval.py --dataset … --videos … --backend gemini
```

## Project layout

```
vidaudit/            # the package: parser, sampler, auditor, backends, report, CLI
eval/                # FineVideo loader, synthetic mutations, captioners, eval runner
notebooks/           # Colab: qwen_smoke (one clip), eval_demo (full eval)
tests/               # pytest — VLM/ffmpeg always mocked
```

## Development

```bash
make check        # lint (ruff) + typecheck (mypy) + tests (pytest) — run before committing
make test
make lint
make typecheck
```

Tests never hit a real VLM API or require a real video — subprocess and SDK
calls are mocked, and a fake VLM backend drives the auditor.

## Limitations

vidaudit verifies static, single-frame claims (objects, entities, attributes).
Action and temporal claims ("the tram *passes* another", event ordering) need
multi-frame reasoning and are out of scope for now — a planned next step is a
separate action-verifier path over densely sampled frames. See
[BACKLOG.md](BACKLOG.md) for the worked example and roadmap.

## License

MIT — see [LICENSE](LICENSE).
