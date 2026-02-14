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

## Meta-Evaluation

The `meta_eval/` module provides a framework for evaluating the judge itself (meta-evaluation) against a human-labeled benchmark. It measures agreement with human judgments, calibration, and bias across multiple judge configurations.

### Benchmark

30 hand-labeled test cases across 9 failure-mode categories:

| Category | Cases | What it tests |
|---|---|---|
| `easy_correct` | 4 | Rephrasing, extra info, numerical equivalence |
| `easy_incorrect` | 3 | Wrong facts, confused events, wrong language behavior |
| `hard_partially_correct` | 5 | Missing precision, subtle technical errors |
| `hallucination_detection` | 4 | False claims in correct context, trap cases |
| `missing_information` | 3 | Omitted details at varying severity |
| `numerical_precision` | 3 | Rounding tolerance, wrong-substance values |
| `entity_confusion` | 3 | Swapped names, anachronistic attributions |
| `verbosity_bias` | 3 | Verbose-but-correct vs. terse-but-incomplete |
| `negation_errors` | 2 | Central claim negated with supporting facts intact |

Each case includes a human verdict (`correct` / `partially_correct` / `incorrect`) and an expected F1 range calibrated to the scoring thresholds.

### Metrics

| Metric | What it measures |
|---|---|
| Verdict accuracy | Exact match between judge and human verdicts |
| F1-in-range accuracy | Judge F1 falls within human-labeled range |
| Cohen's kappa | Chance-corrected inter-rater agreement |
| Mean F1 error | Average distance between judge F1 and expected midpoint |
| False positive rate | Incorrect cases judged as correct |
| False negative rate | Correct cases judged as incorrect |
| Hallucination trap | Whether true additional info is wrongly flagged |

Results are broken down by category and difficulty, with a full 3x3 confusion matrix.

### Running

```bash
# Full comparison: 6 internal configurations x 30 cases
python -m meta_eval.run

# Quick smoke test (3 cases)
python -m meta_eval.run --quick

# Specific configurations only
python -m meta_eval.run --variants baseline,cot+dedup+shuffle,vital

# Test with a different model
python -m meta_eval.run --model openai/gpt-4o

# Test specific failure modes
python -m meta_eval.run --categories negation_errors,hallucination_detection

# Save results to JSON
python -m meta_eval.run --output results.json
```

### Internal Configurations

| Name | Mode | Config |
|---|---|---|
| `baseline` | `long_form` | Default |
| `cot` | `long_form` | CoT extraction |
| `cot+dedup` | `long_form` | CoT + deduplication |
| `cot+dedup+shuffle` | `long_form` | CoT + dedup + position bias mitigation |
| `simple` | `simple` | Single-prompt evaluation |
| `vital` | `vital` | Importance-weighted evaluation |

### Comparing with External Judges

The meta-evaluation can compare correctness-judge head-to-head with other LLM-as-judge libraries on the same benchmark, using the same metrics.

**Supported external judges:**

| Adapter | Library | Install | Evaluation approach |
|---|---|---|---|
| `ragas` | [RAGAS](https://docs.ragas.io/) | `pip install ragas` | Claim-decomposition F1 (FactualCorrectness) |
| `deepeval` | [DeepEval](https://docs.confident-ai.com/) | `pip install deepeval` | G-Eval with CoT and custom criteria |
| `openevals` | [OpenEvals](https://github.com/langchain-ai/openevals) | `pip install openevals` | LLM-as-judge with CORRECTNESS_PROMPT |

```bash
# List which external judges are installed
python -m meta_eval.run --list-external

# Compare internal baseline against RAGAS and DeepEval
python -m meta_eval.run --variants baseline --external ragas,deepeval

# Run all installed external judges
python -m meta_eval.run --variants baseline,cot+dedup+shuffle --external all

# External judges only (skip internal variants)
python -m meta_eval.run --variants none --external ragas,deepeval,openevals

# Override the model used by external adapters
python -m meta_eval.run --variants baseline --external ragas,deepeval --external-model gpt-4o
```

**Writing a custom adapter:**

```python
from meta_eval.adapters import ExternalJudgeAdapter, AdapterResult

class MyJudgeAdapter(ExternalJudgeAdapter):
    name = "my_judge"

    @staticmethod
    def _check_import():
        import my_judge_library  # will be called to check availability

    async def evaluate(self, query: str, expected: str, actual: str) -> AdapterResult:
        # Call your judge here
        score = await my_judge_library.evaluate(query, expected, actual)
        return AdapterResult(
            verdict="correct" if score > 0.8 else "partially_correct" if score > 0.4 else "incorrect",
            score=score,
            reason="...",
        )
```

### Public Benchmark Datasets

In addition to the hand-labeled benchmark, the meta-evaluation framework can load established public datasets from HuggingFace for large-scale comparison.

**Requires:** `pip install datasets`

| Dataset | Source | Cases | What it tests |
|---|---|---|---|
| `judgebench` | [ScalerLab/JudgeBench](https://huggingface.co/datasets/ScalerLab/JudgeBench) | 620 | Objective correctness across knowledge, reasoning, math, coding |
| `llmbar` | [princeton-nlp/LLMBar](https://huggingface.co/datasets/princeton-nlp/LLMBar) | 419 | Instruction-following faithfulness (natural + adversarial) |
| `rewardbench` | [allenai/reward-bench](https://huggingface.co/datasets/allenai/reward-bench) | 2,985 | Chat, reasoning, and safety preference pairs |
| `mtbench` | [lmsys/mt_bench_human_judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) | 3,355 | Multi-turn conversation human pairwise judgments |

Each dataset is automatically downloaded from HuggingFace and adapted from pairwise preference format (response A vs B) into the correctness evaluation format (query, expected, actual with verdict).

```bash
# List available datasets
python -m meta_eval.run_benchmarks --list-datasets

# Quick test: 30 LLMBar cases with internal baseline
python -m meta_eval.run_benchmarks --datasets llmbar --max-cases 30

# Compare judges on multiple datasets
python -m meta_eval.run_benchmarks --datasets llmbar,rewardbench \
    --variants baseline,cot+dedup+shuffle \
    --external ragas,deepeval \
    --max-cases 100

# Full evaluation on all datasets
python -m meta_eval.run_benchmarks --datasets all \
    --variants baseline,simple,vital \
    --external all \
    --max-cases 50 \
    --output benchmark_results.json

# External judges only
python -m meta_eval.run_benchmarks --datasets llmbar \
    --variants none --external ragas,deepeval,openevals,vanilla_llm

# Override models
python -m meta_eval.run_benchmarks --datasets llmbar \
    --model anthropic/claude-sonnet-4-20250514 \
    --external-model gpt-4o-mini \
    --max-cases 30
```

The output includes a ranked comparison table with verdict accuracy, Cohen's kappa, and F1 error for each judge, broken down by category and difficulty.

## References

- VITAL framework: [arXiv:2510.07083](https://arxiv.org/abs/2510.07083)
- G-Eval: [arXiv:2303.16634](https://arxiv.org/abs/2303.16634)
- TREC RAG 2024 AutoNuggetizer: [arXiv:2504.15068](https://arxiv.org/abs/2504.15068)
- PINE position-invariant evaluation: [arXiv:2407.01100](https://arxiv.org/abs/2407.01100)
- D-FActScore entity disambiguation: [arXiv:2402.05629](https://arxiv.org/abs/2402.05629)
- QuanTemp++ numerical fact-checking: [arXiv:2510.22055](https://arxiv.org/abs/2510.22055)
- DecMetrics claim decomposition: [arXiv:2509.04483](https://arxiv.org/abs/2509.04483)
- JudgeBench: [arXiv:2410.12784](https://arxiv.org/abs/2410.12784)
- LLMBar: [arXiv:2310.07641](https://arxiv.org/abs/2310.07641)
- RewardBench: [arXiv:2403.13787](https://arxiv.org/abs/2403.13787)
- MT-Bench: [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)

## License

MIT
