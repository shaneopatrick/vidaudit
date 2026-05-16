# CLAUDE.md

Canonical guide for Claude Code (and other coding assistants) working in this repository.

---

## 1. Project context

**vidaudit** is a Python CLI tool that audits VLM-generated video descriptions for hallucinations. Given a video file and time-coded text descriptions (JSON), it samples frames at each timestamp, decomposes descriptions into verifiable claims (noun phrases, named entities), uses a VLM to check each claim against the actual frame, and produces a structured audit report with grounding scores.

The full project plan, component specs, and build order live in `PLAN.md` — read it before starting any implementation work.

### Core design insight

Descriptions are decomposed into individual verifiable claims and each claim is verified independently via binary VLM questions ("Is [X] visible in this frame?") — NOT free-text comparison of two generated descriptions.

---

## 2. Repo-specific docs to read first

| File | What's in it |
|---|---|
| `PLAN.md` | Full project plan — component specs, interfaces, build order, cut list |
| `README.md` | Quick start, installation, usage examples |
| `BACKLOG.md` | Planned work and deferred improvements |

---

## 3. Repo layout

```
vidaudit/
├── vidaudit/                   # Main package
│   ├── __init__.py
│   ├── cli.py                  # Typer CLI — `audit` and `parse` commands
│   ├── frame_sampler.py        # ffmpeg-based frame extraction
│   ├── description_parser.py   # spaCy NLP — claim extraction from descriptions
│   ├── report.py               # Audit report generation (JSON + Rich terminal)
│   ├── auditors/
│   │   ├── __init__.py
│   │   └── object_audit.py     # Core audit logic — verify claims against frames
│   └── vlm/
│       ├── __init__.py
│       ├── base.py             # Abstract VLM backend interface
│       ├── gemini.py           # Gemini 2.5 Flash backend (default)
│       └── qwen_vl.py          # Qwen2.5-VL local backend (optional, GPU)
├── eval/
│   ├── finevideo_loader.py     # FineVideo dataset loader + synthetic mutations
│   └── run_eval.py             # Evaluation runner — precision, recall, F1
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── fixtures/               # Sample frames, descriptions, expected outputs
│   ├── test_description_parser.py
│   ├── test_frame_sampler.py
│   └── test_object_audit.py
├── notebooks/
│   └── eval_demo.ipynb         # Colab notebook for FineVideo evaluation
├── examples/
│   └── sample_descriptions.json
├── PLAN.md
├── CLAUDE.md
├── README.md
├── pyproject.toml              # Project metadata, deps, tool config (single source of truth)
├── Makefile                    # Common dev commands
└── .gitignore
```

---

## 4. Tech stack

| Component | Choice | Notes |
|---|---|---|
| Python | 3.10+ | Minimum version for `X \| Y` union syntax in type hints |
| Package manager | `uv` | Fast, PEP 621-compliant. `pyproject.toml` is the single config file |
| CLI framework | Typer | With Rich integration for terminal output |
| Data models | Pydantic v2 | `BaseModel` everywhere — never `dataclasses` for structured data |
| Terminal output | Rich | Tables, progress bars, color-coded verdicts |
| NLP | spaCy (`en_core_web_sm`) | Noun phrase extraction and NER for claim decomposition |
| Default VLM | Gemini 2.5 Flash | Via `google-genai` SDK. Free-tier rate-limited |
| Local VLM | Qwen2.5-VL | Via `transformers` + `torch`. Optional, requires GPU |
| Frame extraction | ffmpeg (subprocess) | NOT opencv, NOT decord — keep deps minimal |
| Image handling | Pillow | PIL Images throughout the pipeline |
| Testing | pytest | With `pytest-asyncio` if needed |
| Linting/formatting | Ruff | Format + check, configured in `pyproject.toml` |
| Type checking | mypy | Configured in `pyproject.toml` |

---

## 5. Language & tooling standards

### Python conventions

- `from __future__ import annotations` in every file
- Type hints on all function signatures
- Google-style docstrings on all public functions and classes
- Pydantic `BaseModel` for all structured data (inputs, outputs, intermediate results)
- Abstract base classes for pluggable backends (`VLMBackend` interface)
- f-strings for string formatting — never `.format()` or `%`
- `pathlib.Path` everywhere — never `os.path`
- Import order: stdlib, third-party, local — separated by blank lines

### Package management with uv

`pyproject.toml` is the single source of truth for project metadata, dependencies, and tool configuration. Use `uv` as the package manager.

```bash
# Environment setup
uv venv                                    # Create virtual environment
uv sync                                    # Install all deps from lock file
uv sync --group dev                        # Include dev dependencies

# Dependency management
uv add <package>                           # Add a runtime dependency
uv add --group dev <package>               # Add a dev dependency
uv lock                                    # Regenerate lock file after manual pyproject.toml edits
uv tree                                    # Show dependency tree (check before adding new deps)

# Running commands in the venv
uv run pytest tests/ -v                    # Run tests
uv run ruff format .                       # Format code
uv run ruff check --fix .                  # Lint with auto-fix
uv run mypy vidaudit/                      # Type check
uv run vidaudit audit --help               # Run the CLI
```

The `pyproject.toml` should use PEP 621 metadata format so it remains compatible with `pip install -e .` and other standard tools.

### Makefile targets

```bash
make install        # uv sync + download spaCy model
make test           # uv run pytest tests/ -v
make lint           # uv run ruff format . && uv run ruff check --fix .
make typecheck      # uv run mypy vidaudit/
make check          # lint + typecheck + test (run before every commit)
make demo           # Run audit on example video
make eval           # Run FineVideo evaluation
```

### Ruff configuration (in `pyproject.toml`)

```toml
[tool.ruff]
target-version = "py310"
line-length = 99

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM", "TCH"]

[tool.ruff.lint.isort]
known-first-party = ["vidaudit"]
```

---

## 6. Testing

- **Run tests:** `uv run pytest tests/ -v` or `make test`
- **Mock VLM calls** — tests must never hit real APIs (Gemini, Qwen). Use `unittest.mock.patch` or pytest fixtures.
- **Mock ffmpeg calls** — frame sampler tests should mock subprocess calls, not require real video files.
- **Fixture files** in `tests/fixtures/` for sample frames (small PNGs), descriptions (JSON), and expected outputs.
- **Every new component ships with tests.** Minimum: happy path + one edge case.
- **Pydantic model tests** — verify serialization/deserialization for all data models that cross component boundaries.
- **No flaky tests.** Fix or remove — never `@pytest.mark.skip` to green the build.
- **Run `make check` before every commit.** Lint + typecheck + tests must all pass.

### Test file naming

- `tests/test_<module>.py` — matches the module under test
- Test functions: `test_<behavior>_<scenario>` (e.g., `test_parse_claims_empty_description`)

---

## 7. Security & secrets

- **Never commit API keys.** `.env` files are gitignored. Commit `.env.example` with placeholder values only.
- **Required env vars at runtime:**
  - `GEMINI_API_KEY` — required for default Gemini backend
- **Optional env vars:**
  - `VIDAUDIT_BACKEND` — override VLM backend (`gemini` | `qwen-vl`)
  - `VIDAUDIT_CACHE_DIR` — cache directory for extracted frames (default: `.vidaudit_cache/`)
- **Input validation:** validate external input (JSON descriptions, CLI args) with Pydantic at the boundary. Internal functions trust their callers.
- **Subprocess calls:** never interpolate user input into ffmpeg commands without sanitization. Use list-form `subprocess.run()`, not shell strings.
- **Dependencies:** pin via `uv.lock`. Before adding a new dep, check `uv tree` for existing equivalents.

---

## 8. Git workflow

**Branch prefixes:**
- `feat/` — new capability
- `fix/` — bug fix
- `chore/` — refactor, dep bump, cleanup
- `docs/` — documentation-only changes

**Commit messages:** conventional-commit style (`feat:`, `fix:`, `chore:`, `test:`, `docs:`). Describe the *why* — the diff shows the what.

**Before committing:**
```bash
make check          # lint + typecheck + tests — must pass
```

---

## 9. Key design decisions

These are intentional choices — don't "fix" them:

1. **Claims-based verification, not text comparison.** Descriptions are decomposed into noun phrases/entities and each is verified independently with a binary VLM question.
2. **VLM backends are pluggable** via abstract base class (`VLMBackend`). Default is Gemini 2.5 Flash (free tier).
3. **Frame extraction uses ffmpeg subprocess calls** — not opencv, not decord. Keeps the dependency footprint small and avoids C extension build issues.
4. **Context frames cover the segment span, not a point.** A description covers a time range, so the primary frame is sampled at the segment midpoint and context frames are spread across `[timestamp_start, timestamp_end]` — a claim true only briefly within the span isn't falsely flagged, and this also absorbs motion blur / brief occlusion. When `timestamp_end` is absent the effective end is inferred (next segment's start → video duration for the final segment → capped at `max_segment_span`); if the span collapses to a point, fall back to `t ± context_window`. Inferred ends are recorded in report metadata, never silently fabricated.
5. **All structured data uses Pydantic models** so results serialize cleanly to JSON and validate at boundaries.
6. **spaCy for NLP extraction, not an LLM.** Claim decomposition is deterministic and fast — no need for a second LLM call.
7. **Batch verification** — multiple claims per frame are sent in a single VLM prompt to save API calls.

---

## 10. Pipeline flow

```
Input: video.mp4 + descriptions.json
  │
  ├─ description_parser.py ──► Extract claims (noun phrases, entities) via spaCy
  │
  ├─ frame_sampler.py ──► Extract frames at each timestamp via ffmpeg
  │
  ├─ auditors/object_audit.py ──► For each segment:
  │     │                           1. Verify claims against primary frame (VLM)
  │     │                           2. Check context frames if unsupported
  │     │                           3. Compute grounding score
  │     │
  │     └─ vlm/gemini.py (or qwen_vl.py) ──► Binary verification questions
  │
  └─ report.py ──► Aggregate results → JSON report + Rich terminal summary
```

---

## 11. How to work effectively in this repo

- **Read `PLAN.md` first.** It has component specs, interfaces, edge cases, and build order.
- **Follow the build order** in PLAN.md §Build Order. Each step depends on prior steps.
- **Match existing patterns.** If you see a convention in existing code, follow it.
- **Prefer editing to creating.** Extend existing modules unless the new code is genuinely orthogonal.
- **Don't pad with ceremony.** No trailing summaries, no `# removed` comments, no backwards-compat shims.
- **Default to no comments.** Only comment when the *why* is non-obvious.
- **Confirm before destructive actions** (`git reset --hard`, force push, deleting branches).

---

## 12. System requirements

- Python 3.10+
- ffmpeg (system install — `brew install ffmpeg` / `apt install ffmpeg`)
- ~500MB disk for spaCy `en_core_web_sm` model
- For Qwen backend: CUDA GPU with 16GB+ VRAM (optional, not needed for Gemini)

---

## 13. End-of-turn etiquette

State what changed and what's next in one or two sentences. Skip trailing recaps and don't narrate tool calls.
