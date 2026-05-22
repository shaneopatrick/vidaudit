"""Run vidaudit against the labeled dataset and score it vs the baseline.

This is the project's headline deliverable (DD-15): it shows the
claims-decomposition approach (DD-1) beats the naive "two noisy descriptions"
baseline on *realistic* errors, with the numbers split by subset so a reviewer
can see where accuracy comes from (DD-13).

What it produces:

* **vidaudit vs baseline**, side by side, as precision / recall / F1.
* **Synthetic and real subsets reported separately** — never averaged (DD-13).
* **A confidence-threshold sweep** — the shipped default (DD-12) is *picked*
  here from the best-F1 point, not asserted. The sweep is cheap because the
  verification cache (DD-11) means re-scoring at a new threshold replays cached
  verdicts instead of re-calling the VLM.
* **Extraction recall** — the fraction of mutated spans the parser actually
  surfaced as claims, isolating extraction failures from verification failures
  (DD-13): a claim never extracted can never be flagged, and that is a
  different problem from the VLM getting the verdict wrong.

The scoring is pure and unit-tested. The VLM/frame/captioner wiring is injected
(:func:`make_vidaudit_auditor`, :func:`make_frame_for`) so this module stays
testable without a GPU or network; the canonical run is the Colab notebook.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, computed_field

from eval.finevideo_loader import EvalSample, load_dataset
from vidaudit.description_parser import DescriptionSegment, extract_claims

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from pathlib import Path

    from PIL import Image

    from eval.captioner import Captioner
    from vidaudit.auditors.object_audit import SegmentAuditResult
    from vidaudit.vlm.base import VLMBackend

logger = logging.getLogger(__name__)

_DEFAULT_CONFIDENCE_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_DEFAULT_BASELINE_THRESHOLD = 0.5
_DEFAULT_CONTEXT_WINDOW = 1.0


class Outcome(str, Enum):
    tp = "tp"
    fp = "fp"
    fn = "fn"
    tn = "tn"
    skip = "skip"  # unlabeled (real, not hand-labeled) — excluded from metrics


# ---- scoring primitives (pure) -------------------------------------------


def ground_truth_positive(sample: EvalSample) -> bool | None:
    """Is this sample a known hallucination?

    Synthetic samples derive ground truth from ``mutation_type`` (a mutation
    means a planted hallucination; the clean control is negative). Real samples
    use the hand-label ``real_is_hallucinated`` (``None`` = unlabeled → excluded
    from metrics, DD-13).
    """
    if sample.source == "synthetic":
        return sample.mutation_type is not None
    return sample.real_is_hallucinated


def text_similarity(a: str, b: str) -> float:
    """Token-set Jaccard similarity in [0, 1].

    Deliberately simple and order-independent — this is the *baseline's*
    similarity measure (the "two noisy descriptions" approach DD-1 rejects),
    not vidaudit's. Two empty strings are identical; one empty is disjoint.
    """
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _spans_match(a: str, b: str) -> bool:
    """Loose claim-span match: equal or one contains the other (whole text)."""
    a, b = a.lower().strip(), b.lower().strip()
    return bool(a) and bool(b) and (a == b or a in b or b in a)


def _confusion_outcome(ground_truth: bool, predicted_positive: bool) -> Outcome:
    if ground_truth:
        return Outcome.tp if predicted_positive else Outcome.fn
    return Outcome.fp if predicted_positive else Outcome.tn


def classify_vidaudit(sample: EvalSample, audit: SegmentAuditResult) -> Outcome:
    """Score one vidaudit result against ground truth (claim-level).

    For a known positive with a labeled ``mutated_span``, this is a TP only if
    that specific span is among the flagged claims — vidaudit is held to
    *localizing* the hallucination, not merely flagging the segment. For a
    known negative (clean control), any flagged claim is a false positive.
    """
    gt = ground_truth_positive(sample)
    if gt is None:
        return Outcome.skip

    flagged = [cr.claim.text for cr in audit.claim_results if cr.flagged]
    if gt:
        if sample.mutated_span:
            hit = any(_spans_match(sample.mutated_span, f) for f in flagged)
            return Outcome.tp if hit else Outcome.fn
        # Real positive without a span label → segment-level: any flag counts.
        return Outcome.tp if flagged else Outcome.fn
    return Outcome.fp if flagged else Outcome.tn


def classify_baseline(sample: EvalSample, caption: str, similarity_threshold: float) -> Outcome:
    """Score the text-comparison baseline on one sample.

    The baseline re-captions the frame and flags the segment when the caption
    is too dissimilar from the description. It is segment-level — it has no
    notion of which claim is wrong.
    """
    gt = ground_truth_positive(sample)
    if gt is None:
        return Outcome.skip
    similarity = text_similarity(sample.mutated_description, caption)
    predicted_positive = similarity < similarity_threshold
    return _confusion_outcome(gt, predicted_positive)


def target_span_extracted(sample: EvalSample, audit: SegmentAuditResult) -> bool | None:
    """Was the planted span surfaced as a claim at all? (extraction quality)

    Returns ``None`` for samples without a labeled span (clean controls, real).
    """
    if not sample.mutated_span:
        return None
    return any(_spans_match(sample.mutated_span, cr.claim.text) for cr in audit.claim_results)


# ---- aggregate models -----------------------------------------------------


class ConfusionMatrix(BaseModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, outcome: Outcome) -> None:
        if outcome is Outcome.tp:
            self.tp += 1
        elif outcome is Outcome.fp:
            self.fp += 1
        elif outcome is Outcome.fn:
            self.fn += 1
        elif outcome is Outcome.tn:
            self.tn += 1
        # skip: excluded

    @computed_field  # type: ignore[prop-decorator]
    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


class SweepPoint(BaseModel):
    confidence_threshold: float
    confusion: ConfusionMatrix


class MethodMetrics(BaseModel):
    method: str  # "vidaudit" | "baseline"
    subset: str  # "synthetic" | "real"
    confusion: ConfusionMatrix


class EvalReport(BaseModel):
    n_samples: int
    n_scored: int  # excludes skipped (unlabeled real) samples
    best_confidence_threshold: float
    extraction_recall: float
    sweep: list[SweepPoint]
    metrics: list[MethodMetrics]

    def save_json(self, path: Path, *, indent: int = 2) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=indent))


# ---- orchestration --------------------------------------------------------


def _subset_of(sample: EvalSample) -> str:
    return sample.source


def _sweep_confidence(
    samples: Sequence[EvalSample],
    auditor: Callable[[EvalSample, float], SegmentAuditResult | None],
    grid: Sequence[float],
) -> list[SweepPoint]:
    """Confusion over the synthetic subset at each confidence threshold."""
    points: list[SweepPoint] = []
    for ct in grid:
        cm = ConfusionMatrix()
        for sample in samples:
            if sample.source != "synthetic":
                continue
            audit = auditor(sample, ct)
            if audit is None:
                continue
            cm.add(classify_vidaudit(sample, audit))
        points.append(SweepPoint(confidence_threshold=ct, confusion=cm))
    return points


def _best_threshold(sweep: Sequence[SweepPoint], fallback: float) -> float:
    if not sweep:
        return fallback
    best = max(sweep, key=lambda p: (p.confusion.f1, p.confusion.precision))
    return best.confidence_threshold


def run_eval(
    samples: Sequence[EvalSample],
    auditor: Callable[[EvalSample, float], SegmentAuditResult | None],
    baseline_caption_for: Callable[[EvalSample], str | None],
    *,
    confidence_grid: Sequence[float] = _DEFAULT_CONFIDENCE_GRID,
    baseline_similarity_threshold: float = _DEFAULT_BASELINE_THRESHOLD,
    fallback_confidence_threshold: float = 0.3,
) -> EvalReport:
    """Score vidaudit (with a threshold sweep) and the baseline over ``samples``.

    Args:
        samples: Labeled eval samples (synthetic + real).
        auditor: ``(sample, confidence_threshold) -> SegmentAuditResult | None``.
            Returns ``None`` when a sample can't be audited (no frame). Cheap
            to call repeatedly thanks to the verification cache (DD-11).
        baseline_caption_for: ``(sample) -> caption | None`` — a fresh caption
            of the sample's frame for the text-comparison baseline.
        confidence_grid: Confidence thresholds to sweep for vidaudit.
        baseline_similarity_threshold: Below this Jaccard, the baseline flags.
        fallback_confidence_threshold: Used if the sweep is empty.

    Returns:
        The full :class:`EvalReport`.
    """
    sweep = _sweep_confidence(samples, auditor, confidence_grid)
    best_ct = _best_threshold(sweep, fallback_confidence_threshold)

    # vidaudit metrics at the best threshold, split by subset.
    vid_cm = {"synthetic": ConfusionMatrix(), "real": ConfusionMatrix()}
    base_cm = {"synthetic": ConfusionMatrix(), "real": ConfusionMatrix()}
    extracted_hits = 0
    extracted_total = 0
    n_scored = 0

    for sample in samples:
        subset = _subset_of(sample)
        audit = auditor(sample, best_ct)
        if audit is not None:
            outcome = classify_vidaudit(sample, audit)
            vid_cm[subset].add(outcome)
            if outcome is not Outcome.skip:
                n_scored += 1
            extracted = target_span_extracted(sample, audit)
            if extracted is not None:
                extracted_total += 1
                extracted_hits += int(extracted)

        caption = baseline_caption_for(sample)
        if caption is not None:
            base_cm[subset].add(classify_baseline(sample, caption, baseline_similarity_threshold))

    metrics = [
        MethodMetrics(method="vidaudit", subset="synthetic", confusion=vid_cm["synthetic"]),
        MethodMetrics(method="vidaudit", subset="real", confusion=vid_cm["real"]),
        MethodMetrics(method="baseline", subset="synthetic", confusion=base_cm["synthetic"]),
        MethodMetrics(method="baseline", subset="real", confusion=base_cm["real"]),
    ]
    extraction_recall = extracted_hits / extracted_total if extracted_total else 1.0

    return EvalReport(
        n_samples=len(samples),
        n_scored=n_scored,
        best_confidence_threshold=best_ct,
        extraction_recall=extraction_recall,
        sweep=sweep,
        metrics=metrics,
    )


# ---- terminal rendering ---------------------------------------------------


def render_eval_report(report: EvalReport, console: object | None = None) -> None:
    """Pretty-print the eval report (vidaudit vs baseline, by subset)."""
    from rich.console import Console
    from rich.table import Table

    con = console if console is not None else Console()
    assert isinstance(con, Console)

    con.rule("[bold]vidaudit eval")
    con.print(
        f"samples: [bold]{report.n_samples}[/]  scored: [bold]{report.n_scored}[/]  "
        f"best confidence threshold: [bold]{report.best_confidence_threshold:.2f}[/]  "
        f"extraction recall: [bold]{report.extraction_recall:.2f}[/]\n"
    )

    table = Table(title="Precision / Recall / F1 by method and subset", title_style="bold")
    table.add_column("Method", style="bold")
    table.add_column("Subset")
    table.add_column("TP", justify="right")
    table.add_column("FP", justify="right")
    table.add_column("FN", justify="right")
    table.add_column("TN", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    for m in report.metrics:
        c = m.confusion
        style = "green" if m.method == "vidaudit" else "yellow"
        table.add_row(
            f"[{style}]{m.method}[/]",
            m.subset,
            str(c.tp),
            str(c.fp),
            str(c.fn),
            str(c.tn),
            f"{c.precision:.2f}",
            f"{c.recall:.2f}",
            f"{c.f1:.2f}",
        )
    con.print(table)


# ---- real-run wiring (not unit-tested; Colab-bound) -----------------------


def make_frame_for(
    video_dir: Path,
    *,
    context_window: float = _DEFAULT_CONTEXT_WINDOW,
) -> Callable[[EvalSample], list[Image.Image] | None]:
    """Build a ``frame_for`` that samples frames from ``video_dir/{video_id}.mp4``.

    Samples the primary frame at the segment midpoint with context frames
    across the span (DD-9, simplified: explicit-end midpoint). Returns ``None``
    if the video is missing, the sampler yields nothing, or extraction fails.

    Frame extraction failures (e.g. a FineVideo metadata timestamp that lands
    past the last decodable frame near end-of-video) are caught and skipped:
    this is a batch eval over many videos, so one unreadable frame must not
    abort the whole run. The single-video CLI keeps the stricter behavior of
    raising — there a bad frame is a real problem the user should see.
    """
    from vidaudit.frame_sampler import sample_frames

    def _frame_for(sample: EvalSample) -> list[Image.Image] | None:
        path = video_dir / f"{sample.video_id}.mp4"
        if not path.exists():
            logger.warning("Video missing for %s: %s", sample.video_id, path)
            return None
        if sample.timestamp_end is not None and sample.timestamp_end > sample.timestamp_start:
            span = sample.timestamp_end - sample.timestamp_start
            primary = sample.timestamp_start + span / 2
            window = span / 2
        else:
            primary = sample.timestamp_start
            window = context_window
        try:
            frames = sample_frames(path, [primary], context_window=window).get(primary)
        except RuntimeError as exc:
            logger.warning(
                "Frame extraction failed for %s @ %.1fs — skipping: %s",
                sample.video_id,
                primary,
                exc,
            )
            return None
        return frames or None

    return _frame_for


def make_vidaudit_auditor(
    backend: VLMBackend,
    frame_for: Callable[[EvalSample], list[Image.Image] | None],
    *,
    clean_threshold: float = 0.8,
    partial_threshold: float = 0.4,
) -> Callable[[EvalSample, float], SegmentAuditResult | None]:
    """Build an auditor that parses, samples, and audits a sample.

    The mutated description is parsed into claims and audited against the
    sampled frame at the given confidence threshold. Repeated calls (the
    sweep) reuse cached verifications (DD-11) so only the first pass spends VLM
    quota.
    """
    from vidaudit.auditors.object_audit import audit_segment

    def _audit(sample: EvalSample, confidence_threshold: float) -> SegmentAuditResult | None:
        frames = frame_for(sample)
        if not frames:
            return None
        segment = DescriptionSegment(
            timestamp_start=sample.timestamp_start,
            timestamp_end=sample.timestamp_end,
            description=sample.mutated_description,
            claims=extract_claims(sample.mutated_description),
        )
        return audit_segment(
            segment,
            frames,
            backend,
            confidence_threshold=confidence_threshold,
            clean_threshold=clean_threshold,
            partial_threshold=partial_threshold,
        )

    return _audit


def make_baseline_caption_for(
    captioner: Captioner,
    frame_for: Callable[[EvalSample], list[Image.Image] | None],
) -> Callable[[EvalSample], str | None]:
    """Build a ``baseline_caption_for`` that captions each sample's primary frame."""

    def _caption_for(sample: EvalSample) -> str | None:
        frames = frame_for(sample)
        if not frames:
            return None
        return captioner(frames[0])

    return _caption_for


def _build_backend(name: str) -> VLMBackend:
    if name == "gemini":
        from vidaudit.vlm.gemini import GeminiBackend

        return GeminiBackend()
    if name == "qwen":
        from vidaudit.vlm.qwen_vl import QwenVLBackend

        return QwenVLBackend()
    raise ValueError(f"Unknown backend: {name}")


def _build_captioner(name: str, backend: VLMBackend) -> Captioner:
    from eval.captioner import GeminiCaptioner, qwen_captioner

    if name == "qwen":
        # Reuse the Qwen backend's runner as the captioner (no second load).
        from vidaudit.vlm.qwen_vl import QwenVLBackend

        assert isinstance(backend, QwenVLBackend)
        return qwen_captioner(backend._runner)
    return GeminiCaptioner()


def main(argv: Iterable[str] | None = None) -> int:
    """Colab/CLI entry point. Loads a dataset, runs the eval, writes results.

    Not exercised by unit tests (it shells out to a real backend + videos);
    the scoring it relies on is tested directly. The canonical run is the
    Colab notebook (step 12).
    """
    from dotenv import load_dotenv
    from rich.console import Console

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, help="EvalSample JSON (from finevideo_loader)."
    )
    parser.add_argument("--videos", required=True, help="Directory of {video_id}.mp4 files.")
    parser.add_argument("--backend", default="qwen", choices=["qwen", "gemini"])
    parser.add_argument("--output", default="eval/results/eval_report.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    from pathlib import Path

    samples = load_dataset(Path(args.dataset))
    backend = _build_backend(args.backend)
    frame_for = make_frame_for(Path(args.videos))
    auditor = make_vidaudit_auditor(backend, frame_for)
    captioner = _build_captioner(args.backend, backend)
    baseline_caption_for = make_baseline_caption_for(captioner, frame_for)

    report = run_eval(samples, auditor, baseline_caption_for)

    console = Console()
    render_eval_report(report, console)
    report.save_json(Path(args.output))
    console.print(f"\nResults written to [cyan]{args.output}[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
