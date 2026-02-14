#!/usr/bin/env python3
"""Run the meta-evaluation and print comparison results.

Usage:
    # Run all default internal variants
    python -m meta_eval.run

    # Run specific internal variants
    python -m meta_eval.run --variants baseline,cot+dedup+shuffle,simple,vital

    # Compare against external judges (requires pip install ragas deepeval openevals)
    python -m meta_eval.run --external ragas,deepeval,openevals

    # Run only external judges
    python -m meta_eval.run --variants none --external ragas,deepeval

    # Run internal baseline + all available external judges
    python -m meta_eval.run --variants baseline --external all

    # Run with a specific model override
    python -m meta_eval.run --model openai/gpt-4o

    # Run only specific categories
    python -m meta_eval.run --categories easy_correct,negation_errors

    # Quick smoke test (3 cases)
    python -m meta_eval.run --quick

    # List available external adapters
    python -m meta_eval.run --list-external

    # Save results to JSON
    python -m meta_eval.run --output results.json
"""

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

# Load .env file from project root
_env_file = Path(__file__).resolve().parent.parent / ".env"
try:
    from dotenv import load_dotenv
    if _env_file.exists():
        load_dotenv(_env_file, override=True)
except ImportError:
    pass

from .benchmark import BENCHMARK_CASES
from .runner import (
    JudgeVariant,
    DEFAULT_VARIANTS,
    run_comparison,
)
from .metrics import format_report, format_comparison
from .adapters import ADAPTERS, list_adapters, get_adapter

from correctness_judge import JudgeConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Meta-evaluate correctness-judge against human labels and external judges"
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help=(
            "Comma-separated internal variant names to run (default: all). "
            "Use 'none' to skip internal variants."
        ),
    )
    parser.add_argument(
        "--external",
        type=str,
        default=None,
        help=(
            "Comma-separated external judges to compare: ragas, deepeval, openevals. "
            "Use 'all' for every installed adapter."
        ),
    )
    parser.add_argument(
        "--list-external",
        action="store_true",
        help="List available external adapters and exit",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model for all variants (e.g. openai/gpt-4o)",
    )
    parser.add_argument(
        "--external-model",
        type=str,
        default=None,
        help="Override model for external adapters (e.g. gpt-4o-mini)",
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
        quick_ids = {"easy_correct_01", "hard_partial_01", "easy_incorrect_01"}
        cases = [c for c in cases if c["id"] in quick_ids]
    elif args.categories:
        cats = set(args.categories.split(","))
        cases = [c for c in cases if c["category"] in cats]

    return cases


def select_variants(args) -> list:
    if args.variants and args.variants.lower() == "none":
        return []

    if args.variants:
        names = set(args.variants.split(","))
        variants = [v for v in DEFAULT_VARIANTS if v.name in names]
        if not variants:
            print(f"No matching variants. Available: {[v.name for v in DEFAULT_VARIANTS]}")
            sys.exit(1)
    else:
        variants = list(DEFAULT_VARIANTS)

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


def select_adapters(args) -> list:
    if not args.external:
        return []

    availability = list_adapters()

    if args.external.lower() == "all":
        names = [name for name, avail in availability.items() if avail]
        if not names:
            print("No external adapters installed. Install with:")
            print("  pip install ragas deepeval openevals")
            return []
    else:
        names = args.external.split(",")

    adapters = []
    for name in names:
        name = name.strip()
        if name not in ADAPTERS:
            print(f"Unknown adapter: '{name}'. Available: {list(ADAPTERS.keys())}")
            continue
        if not availability.get(name, False):
            print(f"Adapter '{name}' is not installed. Install with: pip install {name}")
            continue

        kwargs = {}
        if args.external_model:
            kwargs["model"] = args.external_model

        adapters.append(get_adapter(name, **kwargs))

    return adapters


def serialize_report(report):
    """Convert report to JSON-serializable dict."""
    d = asdict(report)
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

    # --list-external: show available adapters and exit
    if args.list_external:
        print("\nExternal LLM-as-Judge Adapters:")
        print(f"{'Name':<15} {'Installed':<12} {'Install command'}")
        print(f"{'-'*15} {'-'*11} {'-'*30}")
        for name, available in list_adapters().items():
            status = "YES" if available else "no"
            print(f"{name:<15} {status:<12} pip install {name}")
        print()
        return

    cases = select_cases(args)
    variants = select_variants(args)
    adapters = select_adapters(args)

    if not variants and not adapters:
        print("Nothing to run. Specify --variants and/or --external.")
        sys.exit(1)

    all_names = [v.name for v in variants] + [a.name for a in adapters]
    print(f"\nRunning meta-evaluation: {len(all_names)} judges x {len(cases)} cases")
    print(f"Judges: {all_names}")
    print()

    reports = await run_comparison(
        variants=variants,
        adapters=adapters,
        cases=cases,
        max_concurrent=args.max_concurrent,
    )

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
                "judges": all_names,
            },
            "reports": [serialize_report(r) for r in reports],
        }
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
