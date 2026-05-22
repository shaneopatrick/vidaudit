"""Tests for the eval runner.

All scoring is pure; the auditor and captioner are fakes, so no VLM, no GPU,
no network. ``SegmentAuditResult`` objects are built directly from Pydantic
models — cheap and deterministic.
"""

from __future__ import annotations

from eval.finevideo_loader import EvalSample, MutationType
from eval.run_eval import (
    ConfusionMatrix,
    EvalReport,
    Outcome,
    classify_baseline,
    classify_vidaudit,
    ground_truth_positive,
    run_eval,
    target_span_extracted,
    text_similarity,
)
from vidaudit.auditors.object_audit import ClaimResult, SegmentAuditResult
from vidaudit.description_parser import Claim, DescriptionSegment
from vidaudit.vlm.base import Verdict, VerificationResult

# ---- builders -------------------------------------------------------------


def _audit_with(flags: dict[str, bool], description: str = "x") -> SegmentAuditResult:
    """A SegmentAuditResult where ``flags`` maps claim text -> flagged?."""
    claim_results: list[ClaimResult] = []
    for text, flagged in flags.items():
        claim = Claim(text=text, claim_type="object", source_description=description)
        verdict: Verdict = "unsupported" if flagged else "supported"
        verification = VerificationResult(claim=text, verdict=verdict, confidence=0.9, evidence="")
        claim_results.append(ClaimResult(claim=claim, verification=verification, flagged=flagged))
    total = len(claim_results) or 1
    supported = sum(1 for cr in claim_results if cr.verification.verdict == "supported")
    grounding = supported / total
    return SegmentAuditResult(
        segment=DescriptionSegment(
            timestamp_start=0.0,
            description=description,
            claims=[cr.claim for cr in claim_results],
        ),
        claim_results=claim_results,
        grounding_score=grounding,
        hallucination_count=sum(1 for f in flags.values() if f),
        verdict="clean" if grounding >= 0.8 else "full_hallucination",
    )


def _mutated(span: str = "cat", description: str = "A cat runs.") -> EvalSample:
    return EvalSample(
        video_id="v",
        timestamp_start=0.0,
        timestamp_end=2.0,
        clean_description="A dog runs.",
        mutated_description=description,
        mutation_type=MutationType.object_swap,
        original_span="dog",
        mutated_span=span,
        source="synthetic",
    )


def _clean(description: str = "A dog runs.") -> EvalSample:
    return EvalSample(
        video_id="v",
        timestamp_start=0.0,
        timestamp_end=2.0,
        clean_description=description,
        mutated_description=description,
        mutation_type=None,
        source="synthetic",
    )


def _real(label: bool | None, caption: str = "A cat sits.") -> EvalSample:
    return EvalSample(
        video_id="v",
        timestamp_start=0.0,
        timestamp_end=2.0,
        clean_description="A dog runs.",
        mutated_description=caption,
        source="real",
        real_is_hallucinated=label,
    )


# ---- text_similarity ------------------------------------------------------


def test_text_similarity_bounds() -> None:
    assert text_similarity("a dog runs", "a dog runs") == 1.0
    assert text_similarity("dog", "cat") == 0.0
    assert text_similarity("", "") == 1.0
    assert text_similarity("dog", "") == 0.0
    assert 0.0 < text_similarity("a red dog", "a blue dog") < 1.0


# ---- ground_truth_positive ------------------------------------------------


def test_ground_truth_positive_by_source() -> None:
    assert ground_truth_positive(_mutated()) is True
    assert ground_truth_positive(_clean()) is False
    assert ground_truth_positive(_real(True)) is True
    assert ground_truth_positive(_real(False)) is False
    assert ground_truth_positive(_real(None)) is None


# ---- classify_vidaudit ----------------------------------------------------


def test_vidaudit_tp_when_target_span_flagged() -> None:
    audit = _audit_with({"cat": True})
    assert classify_vidaudit(_mutated(span="cat"), audit) is Outcome.tp


def test_vidaudit_fn_when_target_span_not_flagged() -> None:
    audit = _audit_with({"cat": False})
    assert classify_vidaudit(_mutated(span="cat"), audit) is Outcome.fn


def test_vidaudit_fn_when_a_different_claim_flagged_not_the_target() -> None:
    # Flagging the wrong claim is not a hit — vidaudit must localize.
    audit = _audit_with({"cat": False, "grass": True})
    assert classify_vidaudit(_mutated(span="cat"), audit) is Outcome.fn


def test_vidaudit_fp_on_clean_control_when_anything_flagged() -> None:
    audit = _audit_with({"dog": True})
    assert classify_vidaudit(_clean(), audit) is Outcome.fp


def test_vidaudit_tn_on_clean_control_when_nothing_flagged() -> None:
    audit = _audit_with({"dog": False})
    assert classify_vidaudit(_clean(), audit) is Outcome.tn


def test_vidaudit_skip_on_unlabeled_real() -> None:
    audit = _audit_with({"cat": True})
    assert classify_vidaudit(_real(None), audit) is Outcome.skip


def test_vidaudit_real_positive_any_flag_is_tp() -> None:
    audit = _audit_with({"cat": True})
    assert classify_vidaudit(_real(True), audit) is Outcome.tp


# ---- classify_baseline ----------------------------------------------------


def test_baseline_tp_when_caption_dissimilar_to_mutated_text() -> None:
    sample = _mutated(description="A cat runs.")
    outcome = classify_baseline(
        sample, caption="totally unrelated words", similarity_threshold=0.5
    )
    assert outcome is Outcome.tp


def test_baseline_fn_when_caption_matches_mutated_text() -> None:
    sample = _mutated(description="A cat runs.")
    outcome = classify_baseline(sample, caption="A cat runs.", similarity_threshold=0.5)
    assert outcome is Outcome.fn


def test_baseline_fp_on_clean_when_caption_dissimilar() -> None:
    sample = _clean(description="A dog runs in the park.")
    outcome = classify_baseline(sample, caption="zebra spaceship ocean", similarity_threshold=0.5)
    assert outcome is Outcome.fp


def test_baseline_skip_on_unlabeled_real() -> None:
    assert classify_baseline(_real(None), "anything", 0.5) is Outcome.skip


# ---- target_span_extracted ------------------------------------------------


def test_target_span_extracted_true_when_present() -> None:
    audit = _audit_with({"cat": False})
    assert target_span_extracted(_mutated(span="cat"), audit) is True


def test_target_span_extracted_false_when_absent() -> None:
    audit = _audit_with({"grass": False})
    assert target_span_extracted(_mutated(span="cat"), audit) is False


def test_target_span_extracted_none_without_span() -> None:
    audit = _audit_with({"dog": False})
    assert target_span_extracted(_clean(), audit) is None


# ---- ConfusionMatrix ------------------------------------------------------


def test_confusion_matrix_metrics() -> None:
    cm = ConfusionMatrix(tp=3, fp=1, fn=1, tn=5)
    assert cm.precision == 0.75
    assert cm.recall == 0.75
    assert abs(cm.f1 - 0.75) < 1e-9


def test_confusion_matrix_empty_metrics_are_zero() -> None:
    cm = ConfusionMatrix()
    assert cm.precision == 0.0
    assert cm.recall == 0.0
    assert cm.f1 == 0.0


def test_confusion_matrix_add_ignores_skip() -> None:
    cm = ConfusionMatrix()
    for outcome in (Outcome.tp, Outcome.tp, Outcome.fp, Outcome.skip, Outcome.tn):
        cm.add(outcome)
    assert (cm.tp, cm.fp, cm.fn, cm.tn) == (2, 1, 0, 1)


# ---- run_eval orchestration -----------------------------------------------


def _sweep_auditor(sample: EvalSample, confidence_threshold: float) -> SegmentAuditResult:
    """Fake auditor with a realistic threshold response.

    The planted hallucination is a strong signal (flagged up to ct=0.7); the
    clean control has only weak noise (flagged only at ct<=0.2). So F1 peaks
    in the middle of the grid — exactly what the sweep should find.
    """
    if sample.mutation_type is not None:
        return _audit_with({sample.mutated_span or "x": confidence_threshold <= 0.7})
    return _audit_with({"tree": confidence_threshold <= 0.2})


def test_run_eval_sweep_picks_best_f1_threshold() -> None:
    samples = [_mutated(), _clean()]
    report = run_eval(
        samples,
        auditor=_sweep_auditor,
        baseline_caption_for=lambda _s: None,
    )
    # Best F1 (=1.0) first achieved at ct=0.3 (target flagged, noise gone).
    assert report.best_confidence_threshold == 0.3
    assert len(report.sweep) == 9


def test_run_eval_splits_synthetic_and_real_metrics() -> None:
    samples = [_mutated(), _clean(), _real(True), _real(None)]

    def auditor(sample: EvalSample, _ct: float) -> SegmentAuditResult:
        if sample.source == "real":
            return _audit_with({"cat": True})  # flags the real positive
        return _sweep_auditor(sample, _ct)

    report = run_eval(samples, auditor, baseline_caption_for=lambda _s: None)

    by_key = {(m.method, m.subset): m.confusion for m in report.metrics}
    syn = by_key[("vidaudit", "synthetic")]
    real = by_key[("vidaudit", "real")]
    assert syn.tp == 1 and syn.tn == 1 and syn.fp == 0  # mutated caught, clean clean
    assert real.tp == 1  # labeled real positive caught
    # The unlabeled real sample is skipped, not scored.
    assert report.n_scored == 3


def test_run_eval_baseline_is_noisier_on_clean_controls_than_vidaudit() -> None:
    """Headline DD-13 result: the text baseline false-positives where vidaudit doesn't."""
    samples = [_clean(description="A dog runs in a green park.")]

    def auditor(sample: EvalSample, _ct: float) -> SegmentAuditResult:
        return _audit_with({"dog": False, "green park": False})  # nothing flagged

    # Baseline caption is accurate-but-differently-worded → low Jaccard → flags.
    report = run_eval(
        samples,
        auditor,
        baseline_caption_for=lambda _s: "a canine sprints across grassy parkland",
    )
    by_key = {(m.method, m.subset): m.confusion for m in report.metrics}
    assert by_key[("vidaudit", "synthetic")].fp == 0
    assert by_key[("baseline", "synthetic")].fp == 1


def test_run_eval_reports_extraction_recall() -> None:
    # One mutated span extracted, one not.
    s1 = _mutated(span="cat")
    s2 = _mutated(span="rabbit")

    def auditor(sample: EvalSample, _ct: float) -> SegmentAuditResult:
        if sample.mutated_span == "cat":
            return _audit_with({"cat": True})  # extracted
        return _audit_with({"grass": True})  # "rabbit" not extracted

    report = run_eval([s1, s2], auditor, baseline_caption_for=lambda _s: None)
    assert report.extraction_recall == 0.5


def test_run_eval_skips_samples_with_no_frame() -> None:
    report = run_eval(
        [_mutated()],
        auditor=lambda _s, _ct: None,  # no frame available
        baseline_caption_for=lambda _s: None,
    )
    assert report.n_scored == 0


# ---- EvalReport I/O -------------------------------------------------------


def test_eval_report_json_round_trip() -> None:
    report = run_eval([_mutated(), _clean()], _sweep_auditor, lambda _s: None)
    dumped = report.model_dump_json()
    loaded = EvalReport.model_validate_json(dumped)
    assert loaded == report
