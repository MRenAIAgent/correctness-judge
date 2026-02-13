#!/usr/bin/env python3
"""Run the meta-evaluation and print comparison results.

Usage:
    # Run all default variants (baseline, cot, cot+dedup, etc.)
    python -m meta_eval.run

    # Run specific variants
    python -m meta_eval.run --variants baseline,cot+dedup+shuffle,simple,vital

    # Run with a specific model
    python -m meta_eval.run --model openai/gpt-4o

    # Run only specific categories
    python -m meta_eval.run --categories easy_correct,negation_errors

    # Quick smoke test (3 cases)
    python -m meta_eval.run --quick

    # Save results to JSON
    python -m meta_eval.run --output results.json
"""

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import asdict

from .benchmark import BENCHMARK_CASES
from .runner import (
    JudgeVariant,
    DEFAULT_VARIANTS,
    run_comparison,
    run_variant,
)
from .metrics import format_report, format_comparison

from correctness_judge import JudgeConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Meta-evaluate the correctness-judge")
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated variant names to run (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model for all variants (e.g. openai/gpt-4o)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated categories to include",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke test with 3 cases (one per verdict)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Max concurrent API calls (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def select_cases(args) -> list:
    cases = BENCHMARK_CASES

    if args.quick:
        # One easy correct, one partial, one incorrect
        quick_ids = {"easy_correct_01", "hard_partial_01", "easy_incorrect_01"}
        cases = [c for c in cases if c["id"] in quick_ids]
    elif args.categories:
        cats = set(args.categories.split(","))
        cases = [c for c in cases if c["category"] in cats]

    return cases


def select_variants(args) -> list:
    if args.variants:
        names = set(args.variants.split(","))
        variants = [v for v in DEFAULT_VARIANTS if v.name in names]
        if not variants:
            print(f"No matching variants. Available: {[v.name for v in DEFAULT_VARIANTS]}")
            sys.exit(1)
    else:
        variants = DEFAULT_VARIANTS

    if args.model:
        variants = [
            JudgeVariant(
                name=v.name,
                model=args.model,
                config=v.config,
                mode=v.mode,
            )
            for v in variants
        ]

    return variants


def serialize_report(report):
    """Convert report to JSON-serializable dict."""
    d = asdict(report)
    # Convert tuples in case_results
    for cr in d.get("case_results", []):
        if isinstance(cr.get("human_f1_range"), (list, tuple)):
            cr["human_f1_range"] = list(cr["human_f1_range"])
    return d


async def main():
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cases = select_cases(args)
    variants = select_variants(args)

    print(f"\nRunning meta-evaluation: {len(variants)} variants x {len(cases)} cases")
    print(f"Variants: {[v.name for v in variants]}")
    print()

    reports = await run_comparison(variants, cases=cases, max_concurrent=args.max_concurrent)

    # Print individual reports
    for report in reports:
        print(format_report(report))

    # Print comparison
    if len(reports) > 1:
        print(format_comparison(reports))

    # Save to JSON
    if args.output:
        data = {
            "meta_evaluation": {
                "num_cases": len(cases),
                "variants": [v.name for v in variants],
            },
            "reports": [serialize_report(r) for r in reports],
        }
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
