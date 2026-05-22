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

Production video-indexing systems (e.g. Moments Lab's MXT-2) generate automated
time-coded descriptions of video content. Those descriptions sometimes
hallucinate — naming objects, people, or landmarks that aren't actually on
screen. Academic work in this space (VideoHallucer, ViBe, MESH) ships
*benchmarks*, not a reusable evaluation tool. `vidaudit` fills that gap with a
small, inspectable, pip-installable auditor.

## Core idea: claims, not text comparison

The naive approach generates a second caption and diffs the two texts — but that
compares two noisy outputs, and errors compound. `vidaudit` instead decomposes
the description into independent claims and verifies each one against the frame
with a binary VLM question. Each check is independent, quantifiable, and carries
its own confidence. (This is the project's central design decision — see
[DESIGN.md](DESIGN.md) DD-1.)

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
optional — see [DESIGN.md](DESIGN.md) DD-9 for how a missing end is resolved):

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
behavior) and the cross-backend comparison is itself an eval result. Gemini is
retained for fast local iteration on machines without a GPU. See
[DESIGN.md](DESIGN.md) DD-16 for the full rationale.

The Qwen backend is GPU-bound; run it in Colab (the dev machine here is a
no-GPU Intel Mac). See [`notebooks/qwen_smoke.ipynb`](notebooks/qwen_smoke.ipynb)
for a one-clip smoke test.

## How it works

```
video.mp4 + descriptions.json
   │
   ├─ description_parser ─► spaCy noun-phrase + NER claim extraction
   ├─ frame_sampler ──────► ffmpeg frame-accurate extraction (DD-8), span-aware (DD-9)
   ├─ auditors/object_audit ─► per-claim binary VLM verification, context-frame rescue
   │     └─ vlm/{qwen_vl,gemini} ─► batched, structured, cached (DD-6, DD-10, DD-11)
   └─ report ─────────────► JSON report + Rich terminal summary
```

## Evaluation

The eval *is* the deliverable for this project (it's a portfolio piece for a
benchmarking-focused role). It is built to be rigorous and reproducible
([DESIGN.md](DESIGN.md) DD-13, DD-15):

- **Baseline comparison.** vidaudit's claims-decomposition is measured against
  the naive "re-caption and diff the two texts" approach it argues against.
- **Plausible synthetic mutations**, not random ones — object swaps to a likely
  co-occurring object (`dog`→`cat`), colour/size changes, and named-entity
  injection. Random swaps are trivially detectable and would inflate the
  metrics. The swap tables are hand-curated and fully auditable.
- **A real-hallucination subset** harvested by captioning frames with a weak
  captioner and keeping its natural errors — synthetic mutations alone don't
  resemble a real VLM error distribution.
- **Subsets reported separately**, never averaged.
- **Extraction quality reported on its own**, so a low F1 is attributable to
  spaCy extraction vs VLM verification — they are different failure modes.
- **Thresholds derived, not asserted.** The shipped confidence/grounding
  defaults are picked from a sweep on eval data (DD-12), not by intuition.

### Results

Run [`notebooks/eval_demo.ipynb`](notebooks/eval_demo.ipynb) on Colab to
reproduce. It builds the labeled dataset from FineVideo, runs both methods, and
prints this table. *(Populate from your run — numbers depend on the FineVideo
sample drawn and the model revision.)*

| Method | Subset | Precision | Recall | F1 |
|---|---|---|---|---|
| **vidaudit** (Qwen2.5-VL-3B) | synthetic | _TBD_ | _TBD_ | _TBD_ |
| text-comparison baseline | synthetic | _TBD_ | _TBD_ | _TBD_ |
| **vidaudit** (Qwen2.5-VL-3B) | real | _TBD_ | _TBD_ | _TBD_ |
| text-comparison baseline | real | _TBD_ | _TBD_ | _TBD_ |

Best confidence threshold (from sweep): _TBD_ · Extraction recall: _TBD_

Locally (no GPU) you can run the same eval with the Gemini backend:

```bash
make eval     # uv run python eval/run_eval.py --dataset … --videos … --backend gemini
```

## Project layout

```
vidaudit/            # the package: parser, sampler, auditor, backends, report, CLI
eval/                # FineVideo loader, synthetic mutations, captioners, eval runner
notebooks/           # Colab: qwen_smoke (one clip), eval_demo (full eval)
tests/               # pytest — VLM/ffmpeg always mocked
PLAN.md  DESIGN.md  BACKLOG.md
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
multi-frame reasoning and are out of scope for now — see [BACKLOG.md](BACKLOG.md)
for the worked example and the planned approach.

## License

MIT — see [LICENSE](LICENSE).
