# correctness-judge

LLM-as-judge for evaluating correctness of AI outputs against reference answers.

## Features

- **Claim decomposition**: Breaks answers into atomic factual claims for granular evaluation
- **Soft probability scoring**: Continuous probability estimates instead of binary verdicts
- **VITAL importance weighting**: Weights claims by importance (vital, okay, not important) based on the VITAL framework (arXiv:2510.07083)
- **Anti-sycophancy prompts**: Evaluation prompts designed to hunt for differences first, reducing false positives
- **Bidirectional verification**: Checks both precision (hallucination detection) and recall (missing information)
- **Structured data support**: Handles both free-text and structured (JSON/dict) comparisons

### v0.2 Improvements

- **Claim deduplication**: Merges overlapping claims before scoring to prevent 15-30% score inflation (arXiv:2504.15068)
- **CoT extraction**: Chain-of-thought claim extraction with topic-guided decomposition (arXiv:2509.04483)
- **Position bias mitigation**: Randomized claim ordering in verification prompts (arXiv:2407.01100)
- **Entity disambiguation**: All extraction prompts require explicit entity naming (arXiv:2402.05629)
- **Numerical comparison**: Deterministic tolerance-based comparison for quantitative claims (arXiv:2510.22055)

## Install

```bash
pip install correctness-judge
```

## Quick Start

### Simple correctness evaluation

```python
import asyncio
from correctness_judge import CorrectnessJudge

judge = CorrectnessJudge()

async def main():
    score = await judge.evaluate(
        query="What is the capital of France?",
        expected="Paris is the capital of France.",
        actual="The capital of France is Paris.",
    )
    print(score.verdict)       # CorrectnessVerdict.CORRECT
    print(score.confidence)    # 0.95
    print(score.is_passing)    # True

asyncio.run(main())
```

### Long-form evaluation with claim decomposition

```python
import asyncio
from correctness_judge import CorrectnessJudge

judge = CorrectnessJudge()

async def main():
    score = await judge.evaluate_long_form(
        query="Describe photosynthesis.",
        expected="Plants convert sunlight into chemical energy using chlorophyll...",
        actual="Photosynthesis is the process by which plants use light energy...",
    )
    print(f"F1: {score.f1}")
    print(f"Precision: {score.precision}, Recall: {score.recall}")
    print(f"Supported: {len(score.supported)}, Contradicted: {len(score.contradicted)}")

asyncio.run(main())
```

### VITAL importance-weighted evaluation

```python
import asyncio
from correctness_judge import VitalCorrectnessJudge

judge = VitalCorrectnessJudge()

async def main():
    score = await judge.evaluate_vital(
        query="What are the side effects of aspirin?",
        expected="Common side effects include stomach irritation and bleeding risk...",
        actual="Aspirin may cause stomach upset and increases bleeding risk...",
    )
    print(f"Vital F1: {score.vital_f1}")
    print(f"Weighted F1: {score.weighted_f1}")
    print(f"Response-level precision: {score.response_level_precision}")
    print(f"Response-level recall: {score.response_level_recall}")
    print(f"Passing: {score.is_passing}")

asyncio.run(main())
```

### Enabling v0.2 improvements

```python
from correctness_judge import CorrectnessJudge, VitalCorrectnessJudge, JudgeConfig

config = JudgeConfig(
    use_cot=True,           # CoT claim extraction
    deduplicate=True,       # Remove overlapping claims
    shuffle_claims=True,    # Randomize verification order
    shuffle_seed=42,        # Reproducible shuffling
    numerical_tolerance=0.05,  # 5% tolerance for numbers
)

judge = CorrectnessJudge(config=config)
vital_judge = VitalCorrectnessJudge(config=config)
```

## JudgeConfig Options

| Option | Default | Description |
|---|---|---|
| `use_cot` | `False` | Use chain-of-thought claim extraction for better coverage |
| `deduplicate` | `False` | Deduplicate overlapping claims before verification |
| `shuffle_claims` | `False` | Randomize claim order to mitigate position bias |
| `shuffle_seed` | `None` | Fixed seed for reproducible shuffling |
| `numerical_tolerance` | `0.05` | Relative tolerance for numerical comparison (5%) |

## Scoring Overview

| Paradigm | Class | Method | Description |
|---|---|---|---|
| Simple | `CorrectnessJudge` | `evaluate()` | Single-prompt verdict with confidence score |
| Decomposed | `CorrectnessJudge` | `evaluate_long_form()` | Claim-level precision, recall, and F1 |
| VITAL | `VitalCorrectnessJudge` | `evaluate_vital()` | Importance-weighted scoring with response-level checks |

## API Reference

| Export | Type | Description |
|---|---|---|
| `CorrectnessJudge` | class | Core judge with simple and decomposed evaluation |
| `VitalCorrectnessJudge` | class | Importance-weighted judge extending CorrectnessJudge |
| `JudgeConfig` | dataclass | Configuration for opt-in improvements |
| `CorrectnessVerdict` | enum | CORRECT, PARTIALLY_CORRECT, INCORRECT |
| `CorrectnessScore` | dataclass | Result from simple evaluation |
| `DecomposedCorrectnessScore` | dataclass | Result from claim-decomposed evaluation |
| `VitalScore` | dataclass | Result from VITAL evaluation |
| `ClaimVerdict` | dataclass | Individual claim verification result |
| `ClaimImportance` | enum | VITAL, OKAY, NOT_IMPORTANT |
| `Nugget` | dataclass | A claim with importance rating |
| `IMPORTANCE_WEIGHTS` | dict | Default weight mapping for importance levels |
| `CORRECTNESS_JUDGE_MODEL` | str | Default model used for evaluation |
| `aggregate_scores()` | static method | Aggregate multiple CorrectnessScore results |

## References

- VITAL framework: [arXiv:2510.07083](https://arxiv.org/abs/2510.07083)
- G-Eval: [arXiv:2303.16634](https://arxiv.org/abs/2303.16634)
- TREC RAG 2024 AutoNuggetizer: [arXiv:2504.15068](https://arxiv.org/abs/2504.15068)
- PINE position-invariant evaluation: [arXiv:2407.01100](https://arxiv.org/abs/2407.01100)
- D-FActScore entity disambiguation: [arXiv:2402.05629](https://arxiv.org/abs/2402.05629)
- QuanTemp++ numerical fact-checking: [arXiv:2510.22055](https://arxiv.org/abs/2510.22055)
- DecMetrics claim decomposition: [arXiv:2509.04483](https://arxiv.org/abs/2509.04483)

## License

MIT
