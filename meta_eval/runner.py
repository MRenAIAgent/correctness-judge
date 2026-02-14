"""Meta-evaluation runner: evaluates judge configurations against the benchmark.

Supports running multiple judge configurations in sequence and producing
a comparison report. Works with both internal correctness-judge variants
and external LLM-as-judge libraries (RAGAS, DeepEval, OpenEvals).
"""

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

from correctness_judge import (
    CorrectnessJudge,
    VitalCorrectnessJudge,
    JudgeConfig,
)

from .adapters import ExternalJudgeAdapter, AdapterResult
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


def _check_api_key():
    """Check that an API key is configured before running evaluations."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if not key and not openai_key:
        print("\n" + "=" * 60)
        print("  ERROR: No API key found!")
        print("=" * 60)
        print()
        print("  Set at least one of these environment variables:")
        print()
        print("    export ANTHROPIC_API_KEY=sk-ant-...")
        print("    export OPENAI_API_KEY=sk-...")
        print()
        print("  Then run again.")
        print("=" * 60 + "\n")
        sys.exit(1)

    if key and len(key) < 10:
        print("\n" + "=" * 60)
        print("  ERROR: ANTHROPIC_API_KEY appears invalid")
        print(f"  (length={len(key)}, expected 100+ chars)")
        print("=" * 60 + "\n")
        sys.exit(1)


async def _test_api_connection(model: str):
    """Run a tiny test call to verify the API key works."""
    try:
        import litellm
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        return True
    except Exception as e:
        error_msg = str(e)
        print("\n" + "=" * 60)
        print("  ERROR: API call failed!")
        print("=" * 60)
        print(f"\n  Model: {model}")
        print(f"  Error: {error_msg[:200]}")
        print()
        if "api-key" in error_msg.lower() or "auth" in error_msg.lower() or "api_key" in error_msg.lower():
            print("  Your API key is invalid or expired.")
            print("  Check: https://console.anthropic.com/settings/keys")
        elif "rate" in error_msg.lower():
            print("  You're being rate limited. Wait a moment and retry.")
        elif "model" in error_msg.lower():
            print(f"  Model '{model}' may not be available on your plan.")
        print()
        print("  Set a valid key:")
        print("    export ANTHROPIC_API_KEY=sk-ant-...")
        print("=" * 60 + "\n")
        return False


# ---------------------------------------------------------------------------
# Internal judge case runner
# ---------------------------------------------------------------------------

async def run_single_case(
    judge,
    case: dict,
    mode: str,
) -> CaseResult:
    """Run one benchmark case through an internal judge and compare to human label."""
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
        # Re-raise auth errors so they bubble up and stop the run
        error_msg = str(e).lower()
        if any(k in error_msg for k in ["api-key", "api_key", "auth", "401", "403", "invalid"]):
            raise
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


# ---------------------------------------------------------------------------
# External adapter case runner
# ---------------------------------------------------------------------------

async def run_single_case_adapter(
    adapter: ExternalJudgeAdapter,
    case: dict,
) -> CaseResult:
    """Run one benchmark case through an external adapter."""
    query = case["query"]
    expected = case["expected"]
    actual = case["actual"]
    human_verdict = case["human_verdict"]
    human_f1_range = tuple(case["human_f1_range"])

    try:
        result: AdapterResult = await adapter.evaluate(
            query=query, expected=expected, actual=actual
        )
        judge_verdict = result.verdict
        judge_f1 = result.score
        judge_confidence = result.score
    except Exception as e:
        # Re-raise auth errors
        error_msg = str(e).lower()
        if any(k in error_msg for k in ["api-key", "api_key", "auth", "401", "403", "invalid"]):
            raise
        logger.error(f"Adapter case {case['id']} failed: {e}")
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


# ---------------------------------------------------------------------------
# Variant runners
# ---------------------------------------------------------------------------

async def run_variant(
    variant: JudgeVariant,
    cases: Optional[List[dict]] = None,
    max_concurrent: int = 5,
) -> MetaEvalReport:
    """Run all benchmark cases for a single internal judge variant."""
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

    case_results = _collect_results(cases, results, variant.name)
    report = build_report(variant.name, case_results, total_time=elapsed)
    return report


async def run_adapter(
    adapter: ExternalJudgeAdapter,
    cases: Optional[List[dict]] = None,
    max_concurrent: int = 5,
) -> MetaEvalReport:
    """Run all benchmark cases for an external adapter."""
    cases = cases or BENCHMARK_CASES

    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_run(case):
        async with semaphore:
            return await run_single_case_adapter(adapter, case)

    logger.info(f"Running external adapter '{adapter.name}' on {len(cases)} cases...")
    start = time.time()

    tasks = [bounded_run(c) for c in cases]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - start

    case_results = _collect_results(cases, results, adapter.name)
    report = build_report(adapter.name, case_results, total_time=elapsed)
    return report


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

async def run_comparison(
    variants: Optional[List[JudgeVariant]] = None,
    adapters: Optional[List[ExternalJudgeAdapter]] = None,
    cases: Optional[List[dict]] = None,
    max_concurrent: int = 5,
) -> List[MetaEvalReport]:
    """Run internal variants and external adapters, return all reports."""
    variants = variants or []
    adapters = adapters or []
    reports = []

    # Pre-flight check: verify API key is set
    _check_api_key()

    # Pre-flight check: test API connection with a tiny call
    if variants:
        model = variants[0].model
    else:
        model = "anthropic/claude-sonnet-4-20250514"

    print(f"\nTesting API connection ({model})...")
    api_ok = await _test_api_connection(model)
    if not api_ok:
        sys.exit(1)
    print("API connection OK.\n")

    for variant in variants:
        try:
            report = await run_variant(variant, cases=cases, max_concurrent=max_concurrent)
            reports.append(report)
            logger.info(
                f"  {variant.name}: accuracy={report.verdict_accuracy:.1%}, "
                f"kappa={report.cohens_kappa:.3f}, time={report.total_time_seconds:.1f}s"
            )
        except Exception as e:
            error_msg = str(e).lower()
            if any(k in error_msg for k in ["api-key", "api_key", "auth", "401", "403"]):
                print(f"\n  FATAL: API authentication failed for '{variant.name}': {e}")
                print("  Stopping. Fix your API key and retry.\n")
                sys.exit(1)
            else:
                print(f"\n  ERROR: Variant '{variant.name}' failed: {e}")
                print("  Skipping this variant.\n")

    for adapter in adapters:
        try:
            report = await run_adapter(adapter, cases=cases, max_concurrent=max_concurrent)
            reports.append(report)
            logger.info(
                f"  {adapter.name}: accuracy={report.verdict_accuracy:.1%}, "
                f"kappa={report.cohens_kappa:.3f}, time={report.total_time_seconds:.1f}s"
            )
        except Exception as e:
            error_msg = str(e).lower()
            if any(k in error_msg for k in ["api-key", "api_key", "auth", "401", "403"]):
                print(f"\n  FATAL: API authentication failed for '{adapter.name}': {e}")
                print("  Stopping. Fix your API key and retry.\n")
                sys.exit(1)
            else:
                print(f"\n  ERROR: Adapter '{adapter.name}' failed: {e}")
                print("  Skipping this adapter.\n")

    if not reports:
        print("\n  No judges completed successfully. Check your API key and try again.\n")
        sys.exit(1)

    return reports


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_results(cases, results, judge_name: str = "") -> List[CaseResult]:
    """Gather results from asyncio.gather, handling exceptions."""
    case_results = []
    error_count = 0

    for i, r in enumerate(results):
        if isinstance(r, Exception):
            error_count += 1
            # Re-raise auth errors immediately
            error_msg = str(r).lower()
            if any(k in error_msg for k in ["api-key", "api_key", "auth", "401", "403", "invalid"]):
                raise r
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

    # If ALL cases failed, something is fundamentally wrong
    if error_count == len(cases):
        first_error = next((r for r in results if isinstance(r, Exception)), None)
        print(f"\n  FATAL: All {len(cases)} cases failed for '{judge_name}'!")
        print(f"  First error: {first_error}")
        print("  This usually means your API key is invalid or the model is unavailable.")
        print("  Check: export ANTHROPIC_API_KEY=sk-ant-...\n")
        sys.exit(1)
    elif error_count > 0:
        pct = error_count / len(cases) * 100
        logger.warning(
            f"  {judge_name}: {error_count}/{len(cases)} cases failed ({pct:.0f}%) — "
            f"results may be unreliable"
        )

    return case_results
