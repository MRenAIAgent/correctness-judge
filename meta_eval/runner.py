"""Meta-evaluation runner: evaluates judge configurations against the benchmark.

Supports running multiple judge configurations in sequence and producing
a comparison report.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from correctness_judge import (
    CorrectnessJudge,
    VitalCorrectnessJudge,
    JudgeConfig,
)

from .benchmark import BENCHMARK_CASES
from .metrics import CaseResult, MetaEvalReport, build_report

logger = logging.getLogger(__name__)


@dataclass
class JudgeVariant:
    """A named judge configuration to evaluate."""

    name: str
    model: str
    config: JudgeConfig
    mode: str = "long_form"  # "simple", "long_form", or "vital"


# Pre-defined configurations for comparison
DEFAULT_VARIANTS = [
    JudgeVariant(
        name="baseline",
        model="anthropic/claude-sonnet-4-20250514",
        config=JudgeConfig(),
        mode="long_form",
    ),
    JudgeVariant(
        name="cot",
        model="anthropic/claude-sonnet-4-20250514",
        config=JudgeConfig(use_cot=True),
        mode="long_form",
    ),
    JudgeVariant(
        name="cot+dedup",
        model="anthropic/claude-sonnet-4-20250514",
        config=JudgeConfig(use_cot=True, deduplicate=True),
        mode="long_form",
    ),
    JudgeVariant(
        name="cot+dedup+shuffle",
        model="anthropic/claude-sonnet-4-20250514",
        config=JudgeConfig(
            use_cot=True, deduplicate=True, shuffle_claims=True, shuffle_seed=42
        ),
        mode="long_form",
    ),
    JudgeVariant(
        name="simple",
        model="anthropic/claude-sonnet-4-20250514",
        config=JudgeConfig(),
        mode="simple",
    ),
    JudgeVariant(
        name="vital",
        model="anthropic/claude-sonnet-4-20250514",
        config=JudgeConfig(),
        mode="vital",
    ),
]


def _verdict_from_decomposed(score) -> str:
    """Extract verdict string from a DecomposedCorrectnessScore."""
    return score.verdict.value


def _verdict_from_vital(score) -> str:
    """Extract verdict string from a VitalScore."""
    return score.verdict.value


def _f1_from_decomposed(score) -> float:
    """Extract F1 from a DecomposedCorrectnessScore."""
    return score.f1


def _f1_from_vital(score) -> float:
    """Extract vital F1 from a VitalScore."""
    return score.vital_f1


async def run_single_case(
    judge,
    case: dict,
    mode: str,
) -> CaseResult:
    """Run one benchmark case through the judge and compare to human label."""
    query = case["query"]
    expected = case["expected"]
    actual = case["actual"]
    human_verdict = case["human_verdict"]
    human_f1_range = tuple(case["human_f1_range"])

    try:
        if mode == "simple":
            score = await judge.evaluate(query=query, expected=expected, actual=actual)
            judge_verdict = score.verdict.value
            judge_f1 = score.confidence  # Use confidence as proxy for F1
            judge_confidence = score.confidence
        elif mode == "vital":
            score = await judge.evaluate_vital(
                query=query, expected=expected, actual=actual
            )
            judge_verdict = _verdict_from_vital(score)
            judge_f1 = _f1_from_vital(score)
            judge_confidence = score.vital_f1
        else:  # long_form
            score = await judge.evaluate_long_form(
                query=query, expected=expected, actual=actual
            )
            judge_verdict = _verdict_from_decomposed(score)
            judge_f1 = _f1_from_decomposed(score)
            judge_confidence = score.soft_f1 if score.soft_f1 is not None else score.f1
    except Exception as e:
        logger.error(f"Case {case['id']} failed: {e}")
        judge_verdict = "incorrect"
        judge_f1 = 0.0
        judge_confidence = 0.0

    f1_in_range = (
        judge_f1 is not None
        and human_f1_range[0] <= judge_f1 <= human_f1_range[1]
    )

    return CaseResult(
        case_id=case["id"],
        category=case["category"],
        difficulty=case["difficulty"],
        human_verdict=human_verdict,
        judge_verdict=judge_verdict,
        human_f1_range=human_f1_range,
        judge_f1=judge_f1,
        judge_confidence=judge_confidence,
        f1_in_range=f1_in_range,
        verdict_match=(human_verdict == judge_verdict),
    )


async def run_variant(
    variant: JudgeVariant,
    cases: Optional[List[dict]] = None,
    max_concurrent: int = 5,
) -> MetaEvalReport:
    """Run all benchmark cases for a single judge variant."""
    cases = cases or BENCHMARK_CASES

    if variant.mode == "vital":
        judge = VitalCorrectnessJudge(model=variant.model, config=variant.config)
    else:
        judge = CorrectnessJudge(model=variant.model, config=variant.config)

    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_run(case):
        async with semaphore:
            return await run_single_case(judge, case, variant.mode)

    logger.info(f"Running variant '{variant.name}' ({variant.mode}) on {len(cases)} cases...")
    start = time.time()

    tasks = [bounded_run(c) for c in cases]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - start

    # Handle exceptions
    case_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error(f"Case {cases[i]['id']} raised exception: {r}")
            case_results.append(
                CaseResult(
                    case_id=cases[i]["id"],
                    category=cases[i]["category"],
                    difficulty=cases[i]["difficulty"],
                    human_verdict=cases[i]["human_verdict"],
                    judge_verdict="incorrect",
                    human_f1_range=tuple(cases[i]["human_f1_range"]),
                    judge_f1=0.0,
                    judge_confidence=0.0,
                    f1_in_range=False,
                    verdict_match=False,
                )
            )
        else:
            case_results.append(r)

    report = build_report(variant.name, case_results, total_time=elapsed)
    return report


async def run_comparison(
    variants: Optional[List[JudgeVariant]] = None,
    cases: Optional[List[dict]] = None,
    max_concurrent: int = 5,
) -> List[MetaEvalReport]:
    """Run multiple judge variants sequentially and return all reports."""
    variants = variants or DEFAULT_VARIANTS
    reports = []

    for variant in variants:
        report = await run_variant(variant, cases=cases, max_concurrent=max_concurrent)
        reports.append(report)
        logger.info(
            f"  {variant.name}: accuracy={report.verdict_accuracy:.1%}, "
            f"kappa={report.cohens_kappa:.3f}, time={report.total_time_seconds:.1f}s"
        )

    return reports
