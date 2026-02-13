"""Meta-evaluation metrics for measuring judge quality.

Computes agreement, calibration, and bias metrics comparing judge output
against human-labeled ground truth.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


VERDICT_MAP = {"correct": 2, "partially_correct": 1, "incorrect": 0}
VERDICT_LABELS = ["incorrect", "partially_correct", "correct"]


@dataclass
class CaseResult:
    case_id: str
    category: str
    difficulty: str
    human_verdict: str
    judge_verdict: str
    human_f1_range: Tuple[float, float]
    judge_f1: Optional[float] = None
    judge_confidence: Optional[float] = None
    f1_in_range: bool = False
    verdict_match: bool = False


@dataclass
class MetaEvalReport:
    """Aggregated meta-evaluation results for a single judge configuration."""

    config_name: str
    total_cases: int = 0
    # Agreement metrics
    verdict_accuracy: float = 0.0
    verdict_accuracy_strict: float = 0.0
    f1_range_accuracy: float = 0.0
    cohens_kappa: float = 0.0
    # Per-category accuracy
    category_accuracy: Dict[str, float] = field(default_factory=dict)
    # Per-difficulty accuracy
    difficulty_accuracy: Dict[str, float] = field(default_factory=dict)
    # Calibration
    mean_f1_error: float = 0.0
    mean_confidence_error: float = 0.0
    # Bias indicators
    false_positive_rate: float = 0.0  # Incorrect judged as correct
    false_negative_rate: float = 0.0  # Correct judged as incorrect
    hallucination_trap_passed: bool = False  # case hallucination_03
    # Detailed results
    case_results: List[CaseResult] = field(default_factory=list)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Timing
    total_time_seconds: float = 0.0
    avg_time_per_case: float = 0.0


def compute_verdict_accuracy(results: List[CaseResult]) -> float:
    """Exact match accuracy between judge and human verdicts."""
    if not results:
        return 0.0
    matches = sum(1 for r in results if r.verdict_match)
    return matches / len(results)


def compute_verdict_accuracy_lenient(results: List[CaseResult]) -> float:
    """Lenient accuracy: off-by-one verdicts count as half-correct.

    correct vs partially_correct = 0.5 credit
    partially_correct vs incorrect = 0.5 credit
    correct vs incorrect = 0 credit
    """
    if not results:
        return 0.0
    score = 0.0
    for r in results:
        h = VERDICT_MAP.get(r.human_verdict, 0)
        j = VERDICT_MAP.get(r.judge_verdict, 0)
        diff = abs(h - j)
        if diff == 0:
            score += 1.0
        elif diff == 1:
            score += 0.5
    return score / len(results)


def compute_f1_range_accuracy(results: List[CaseResult]) -> float:
    """Fraction of cases where judge F1 falls within human-labeled range."""
    eligible = [r for r in results if r.judge_f1 is not None]
    if not eligible:
        return 0.0
    in_range = sum(1 for r in eligible if r.f1_in_range)
    return in_range / len(eligible)


def compute_cohens_kappa(results: List[CaseResult]) -> float:
    """Cohen's kappa for inter-rater agreement between judge and human.

    Corrects for chance agreement. Range: -1 to 1.
    1 = perfect agreement, 0 = chance, < 0 = worse than chance.
    """
    if not results:
        return 0.0

    labels = list(VERDICT_MAP.keys())
    n = len(results)

    # Build confusion counts
    matrix = {h: {j: 0 for j in labels} for h in labels}
    for r in results:
        h = r.human_verdict if r.human_verdict in labels else "incorrect"
        j = r.judge_verdict if r.judge_verdict in labels else "incorrect"
        matrix[h][j] += 1

    # Observed agreement
    p_o = sum(matrix[l][l] for l in labels) / n

    # Expected agreement
    p_e = 0.0
    for l in labels:
        h_count = sum(matrix[l][j] for j in labels)
        j_count = sum(matrix[h][l] for h in labels)
        p_e += (h_count / n) * (j_count / n)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def compute_confusion_matrix(results: List[CaseResult]) -> Dict[str, Dict[str, int]]:
    """3x3 confusion matrix: human verdict (rows) x judge verdict (cols)."""
    labels = list(VERDICT_MAP.keys())
    matrix = {h: {j: 0 for j in labels} for h in labels}
    for r in results:
        h = r.human_verdict if r.human_verdict in labels else "incorrect"
        j = r.judge_verdict if r.judge_verdict in labels else "incorrect"
        matrix[h][j] += 1
    return matrix


def compute_category_accuracy(results: List[CaseResult]) -> Dict[str, float]:
    """Verdict accuracy broken down by benchmark category."""
    cats: Dict[str, List[CaseResult]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)
    return {cat: compute_verdict_accuracy(rs) for cat, rs in sorted(cats.items())}


def compute_difficulty_accuracy(results: List[CaseResult]) -> Dict[str, float]:
    """Verdict accuracy broken down by difficulty level."""
    diffs: Dict[str, List[CaseResult]] = {}
    for r in results:
        diffs.setdefault(r.difficulty, []).append(r)
    return {d: compute_verdict_accuracy(rs) for d, rs in sorted(diffs.items())}


def compute_false_positive_rate(results: List[CaseResult]) -> float:
    """Rate of human=incorrect cases judged as correct."""
    incorrect_cases = [r for r in results if r.human_verdict == "incorrect"]
    if not incorrect_cases:
        return 0.0
    false_pos = sum(1 for r in incorrect_cases if r.judge_verdict == "correct")
    return false_pos / len(incorrect_cases)


def compute_false_negative_rate(results: List[CaseResult]) -> float:
    """Rate of human=correct cases judged as incorrect."""
    correct_cases = [r for r in results if r.human_verdict == "correct"]
    if not correct_cases:
        return 0.0
    false_neg = sum(1 for r in correct_cases if r.judge_verdict == "incorrect")
    return false_neg / len(correct_cases)


def compute_mean_f1_error(results: List[CaseResult]) -> float:
    """Mean absolute error between judge F1 and midpoint of human F1 range."""
    eligible = [r for r in results if r.judge_f1 is not None]
    if not eligible:
        return 0.0
    total = 0.0
    for r in eligible:
        midpoint = (r.human_f1_range[0] + r.human_f1_range[1]) / 2
        total += abs(r.judge_f1 - midpoint)
    return total / len(eligible)


def build_report(config_name: str, results: List[CaseResult], total_time: float = 0.0) -> MetaEvalReport:
    """Compile all metrics into a single report."""
    confusion = compute_confusion_matrix(results)
    hallucination_trap = next(
        (r for r in results if r.case_id == "hallucination_03"), None
    )

    return MetaEvalReport(
        config_name=config_name,
        total_cases=len(results),
        verdict_accuracy=compute_verdict_accuracy(results),
        verdict_accuracy_strict=compute_verdict_accuracy(results),
        f1_range_accuracy=compute_f1_range_accuracy(results),
        cohens_kappa=compute_cohens_kappa(results),
        category_accuracy=compute_category_accuracy(results),
        difficulty_accuracy=compute_difficulty_accuracy(results),
        mean_f1_error=compute_mean_f1_error(results),
        false_positive_rate=compute_false_positive_rate(results),
        false_negative_rate=compute_false_negative_rate(results),
        hallucination_trap_passed=(
            hallucination_trap is not None and hallucination_trap.verdict_match
        ),
        case_results=results,
        confusion_matrix=confusion,
        total_time_seconds=total_time,
        avg_time_per_case=total_time / len(results) if results else 0.0,
    )


def format_report(report: MetaEvalReport) -> str:
    """Format a report as a human-readable string."""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  META-EVALUATION REPORT: {report.config_name}")
    lines.append(f"{'='*70}")
    lines.append(f"  Cases: {report.total_cases}  |  Time: {report.total_time_seconds:.1f}s  |  Avg: {report.avg_time_per_case:.1f}s/case")
    lines.append("")

    # Agreement
    lines.append("  AGREEMENT METRICS")
    lines.append(f"  {'Verdict accuracy (exact):':<35} {report.verdict_accuracy:.1%}")
    lines.append(f"  {'F1-in-range accuracy:':<35} {report.f1_range_accuracy:.1%}")
    kappa_label = "Cohen's kappa:"
    lines.append(f"  {kappa_label:<35} {report.cohens_kappa:.3f}")
    lines.append("")

    # Bias
    lines.append("  BIAS INDICATORS")
    lines.append(f"  {'False positive rate:':<35} {report.false_positive_rate:.1%}")
    lines.append(f"  {'False negative rate:':<35} {report.false_negative_rate:.1%}")
    lines.append(f"  {'Hallucination trap passed:':<35} {'YES' if report.hallucination_trap_passed else 'NO'}")
    lines.append("")

    # Calibration
    lines.append("  CALIBRATION")
    lines.append(f"  {'Mean F1 error:':<35} {report.mean_f1_error:.3f}")
    lines.append("")

    # By category
    lines.append("  ACCURACY BY CATEGORY")
    for cat, acc in report.category_accuracy.items():
        lines.append(f"    {cat:<35} {acc:.1%}")
    lines.append("")

    # By difficulty
    lines.append("  ACCURACY BY DIFFICULTY")
    for diff, acc in report.difficulty_accuracy.items():
        lines.append(f"    {diff:<35} {acc:.1%}")
    lines.append("")

    # Confusion matrix
    lines.append("  CONFUSION MATRIX (rows=human, cols=judge)")
    labels = ["correct", "partially_correct", "incorrect"]
    header = f"  {'':>20}" + "".join(f"{l:>18}" for l in labels)
    lines.append(header)
    for h in labels:
        row = f"  {h:>20}"
        for j in labels:
            row += f"{report.confusion_matrix.get(h, {}).get(j, 0):>18}"
        lines.append(row)
    lines.append("")

    # Per-case details
    lines.append("  CASE DETAILS")
    lines.append(f"  {'ID':<25} {'Human':<18} {'Judge':<18} {'F1':>6} {'Range':>12} {'Match'}")
    lines.append(f"  {'-'*25} {'-'*17} {'-'*17} {'-'*6} {'-'*12} {'-'*5}")
    for r in report.case_results:
        f1_str = f"{r.judge_f1:.3f}" if r.judge_f1 is not None else "  N/A"
        range_str = f"[{r.human_f1_range[0]:.2f},{r.human_f1_range[1]:.2f}]"
        match_icon = "OK" if r.verdict_match else "MISS"
        lines.append(
            f"  {r.case_id:<25} {r.human_verdict:<18} {r.judge_verdict:<18} {f1_str:>6} {range_str:>12} {match_icon}"
        )
    lines.append(f"{'='*70}\n")
    return "\n".join(lines)


def format_comparison(reports: List[MetaEvalReport]) -> str:
    """Format a side-by-side comparison of multiple judge configurations."""
    if not reports:
        return "No reports to compare."

    lines = []
    lines.append(f"\n{'='*90}")
    lines.append("  JUDGE COMPARISON")
    lines.append(f"{'='*90}")
    lines.append("")

    # Header
    col_w = 18
    header = f"  {'Metric':<35}"
    for r in reports:
        name = r.config_name[:col_w - 2]
        header += f"{name:>{col_w}}"
    lines.append(header)
    lines.append(f"  {'-'*35}" + f"{'-'*col_w}" * len(reports))

    # Metrics rows
    metrics = [
        ("Verdict accuracy", lambda r: f"{r.verdict_accuracy:.1%}"),
        ("F1-in-range accuracy", lambda r: f"{r.f1_range_accuracy:.1%}"),
        ("Cohen's kappa", lambda r: f"{r.cohens_kappa:.3f}"),
        ("Mean F1 error", lambda r: f"{r.mean_f1_error:.3f}"),
        ("False positive rate", lambda r: f"{r.false_positive_rate:.1%}"),
        ("False negative rate", lambda r: f"{r.false_negative_rate:.1%}"),
        ("Hallucination trap", lambda r: "PASS" if r.hallucination_trap_passed else "FAIL"),
        ("Avg time/case (s)", lambda r: f"{r.avg_time_per_case:.1f}"),
    ]

    for name, fn in metrics:
        row = f"  {name:<35}"
        for r in reports:
            row += f"{fn(r):>{col_w}}"
        lines.append(row)

    lines.append("")

    # Category breakdown
    all_cats = sorted(set(c for r in reports for c in r.category_accuracy))
    if all_cats:
        lines.append(f"  {'CATEGORY BREAKDOWN'}")
        lines.append(f"  {'-'*35}" + f"{'-'*col_w}" * len(reports))
        for cat in all_cats:
            row = f"  {cat:<35}"
            for r in reports:
                acc = r.category_accuracy.get(cat, 0.0)
                row += f"{acc:.1%}".rjust(col_w)
            lines.append(row)

    lines.append("")

    # Difficulty breakdown
    all_diffs = ["easy", "medium", "hard"]
    lines.append(f"  {'DIFFICULTY BREAKDOWN'}")
    lines.append(f"  {'-'*35}" + f"{'-'*col_w}" * len(reports))
    for diff in all_diffs:
        row = f"  {diff:<35}"
        for r in reports:
            acc = r.difficulty_accuracy.get(diff, 0.0)
            row += f"{acc:.1%}".rjust(col_w)
        lines.append(row)

    lines.append(f"\n{'='*90}\n")
    return "\n".join(lines)
