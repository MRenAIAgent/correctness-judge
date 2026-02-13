# LLM-as-Judge Deep Research

A comprehensive analysis of the LLM-as-judge landscape comparing our correctness-judge approach with the state of the art. This document identifies gaps and improvement opportunities grounded in recent academic research.

## 1. State of the Art

### 1.1 Evaluation Paradigms

The LLM-as-judge field encompasses four main paradigms:

- **Pointwise**: Single response scored against criteria (our `evaluate()`)
- **Pairwise**: Two responses compared head-to-head (MT-Bench, Chatbot Arena)
- **Reference-based**: Response compared to ground truth (our `evaluate_long_form()`, `evaluate_vital()`)
- **Reference-free**: Response scored without ground truth (RAGAS, TruLens)

Our implementation focuses on reference-based evaluation, which is the strongest approach when ground truth is available.

### 1.2 Key Frameworks

| Framework | Type | Key Innovation |
|-----------|------|----------------|
| FActScore (arXiv:2305.14251) | Precision-only | Atomic fact decomposition against knowledge source |
| SAFE (arXiv:2403.18802) | Precision-only | Multi-step search-augmented fact verification |
| RAGAS (arXiv:2309.15217) | Reference-free | RAG triad: context relevance, faithfulness, answer relevance |
| ARES (arXiv:2311.09476) | Reference-free | Fine-tuned lightweight judges with PPI calibration |
| MiniCheck (arXiv:2404.10774) | Verification | 770M model matching GPT-4 at 400x lower cost |
| VITAL (arXiv:2510.07083) | Importance-weighted | Query-anchored claim importance scoring |
| DeepEval | Multi-metric | G-Eval CoT-guided evaluation, 50+ metrics |
| TruLens | Observability | RAG triad with OpenTelemetry tracing |

### 1.3 Known Failure Modes

- **Sycophancy**: LLMs tend to agree rather than find errors (arXiv:2601.03263)
- **Position bias**: Claims listed first/last in prompts receive different treatment (arXiv:2406.07791)
- **Verbosity bias**: Longer responses rated higher regardless of quality (arXiv:2505.19477)
- **Self-enhancement bias**: Models rate their own outputs higher (arXiv:2406.12624)
- **Entity ambiguity**: Individual claims correct but combined non-factual (arXiv:2402.05629)

## 2. Our Approach vs. the Field

### 2.1 Comparison Matrix

| Capability | correctness-judge | RAGAS | DeepEval | ARES | FActScore |
|---|---|---|---|---|---|
| Claim decomposition | Yes | Partial | Yes | No | Yes |
| Importance weighting (VITAL) | Yes | No | No | No | No |
| Bidirectional P/R/F1 | Yes | No | No | No | No (precision only) |
| Soft probability scoring | Yes | No | No | No | No |
| Anti-sycophancy prompting | Yes | No | No | No | No |
| Self-consistency guard | Yes | No | No | No | No |
| Multi-judge ensemble | No | No | No | No | No |
| Calibration | No | No | No | Yes (PPI) | No |
| Cost optimization | No | No | No | Yes | No |
| Claim deduplication | Yes (v0.2) | No | No | No | No |

### 2.2 Strengths

- **Bidirectional verification**: Both recall (expected vs actual) and precision (actual vs expected)
- **VITAL importance weighting**: Query-anchored claim importance with configurable weights
- **Soft probability scoring**: Continuous P(true|reference) per claim
- **Anti-sycophancy design**: "Hunt for mismatches" framing + programmatic consistency guard
- **Structured data support**: Dedicated evaluation path for JSON/dict responses

### 2.3 Gaps Identified

1. No claim deduplication -- overlapping claims inflate scores (TREC RAG 2024, arXiv:2504.15068)
2. No verifiability filtering -- opinions mixed with facts (VeriScore, arXiv:2406.19276)
3. Batch verification in single prompt -- position bias (SAFE, arXiv:2403.18802)
4. Uncalibrated probability scores -- raw LLM probs don't match true rates
5. Single evaluation strategy -- no ensembling across prompt variants
6. No cost optimization -- every call uses full model

## 3. Improvement Roadmap

### Phase 1: Quick Wins (Implemented in v0.2)

| # | Improvement | Paper Reference | Impact |
|---|-------------|-----------------|--------|
| 1 | Claim deduplication | TREC RAG 2024 (arXiv:2504.15068), GINGER (arXiv:2503.18174) | Eliminates 15-30% score inflation from overlapping claims |
| 2 | Randomize claim order | Position bias studies (arXiv:2406.07791, arXiv:2407.01100) | Mitigates primacy/recency bias in batch verification |
| 3 | CoT claim extraction | G-Eval/DeepEval, DecMetrics (arXiv:2509.04483) | Improves claim completeness and reduces overlap |
| 4 | Entity disambiguation | D-FActScore (arXiv:2402.05629) | Prevents 10%+ factuality overestimation |
| 5 | Numerical claim handling | QuanTemp++ (arXiv:2510.22055) | Deterministic comparison for quantitative claims |

### Phase 2: Medium Investments

| # | Improvement | Paper Reference | Impact |
|---|-------------|-----------------|--------|
| 6 | Verifiability classification | VeriScore (arXiv:2406.19276) | Reduces false negatives from subjective claims |
| 7 | Multi-strategy ensemble | SE-Jury (arXiv:2505.20854) | 29-140% improvement in human correlation |
| 8 | Claim-level confidence | Neither Valid nor Reliable (arXiv:2508.18076) | Separates judge confidence from claim probability |
| 9 | Edge case routing | Claimify (arXiv:2502.10855) | Better handling of short/list/factoid answers |

### Phase 3: Larger Investments

| # | Improvement | Paper Reference | Impact |
|---|-------------|-----------------|--------|
| 10 | Cheap verification tier | MiniCheck (arXiv:2404.10774), Trust or Escalate (arXiv:2407.18370) | 60-80% cost reduction |
| 11 | Confidence calibration | ARES PPI (arXiv:2311.09476), CalibraEval (arXiv:2410.15393) | Makes soft scores interpretable |
| 12 | Individual claim verification | SAFE (arXiv:2403.18802) | Eliminates cross-contamination in batch verification |

## 4. Claim Decomposition Analysis

### 4.1 State of the Art

The "Decompose-Then-Verify" paradigm is standard but has known failure modes (arXiv:2411.02400):

- **Too fine-grained**: Trivial facts dominate the score
- **Too coarse-grained**: Misses specific errors
- **Overlapping claims**: Inflate counts unfairly
- **Entity ambiguity**: Individually correct claims combine into non-factual text

### 4.2 Best Practices from Research

1. **Claimify** (arXiv:2502.10855): Extract claims only with high confidence in interpretation
2. **DecompScore** (arXiv:2403.11903): Measure decomposition quality separately from factuality
3. **DecMetrics** (arXiv:2509.04483): COMPLETENESS, CORRECTNESS, and SEMANTIC ENTROPY metrics
4. **MedScore** (arXiv:2505.18452): Domain-adapted decomposition extracts 3x more valid facts
5. **SUCEA** (arXiv:2506.04583): Iterative evidence retrieval with claim editing

## 5. Bias Mitigation Research

### 5.1 Position Bias

- PINE (arXiv:2407.01100): Bidirectional attention between documents, 8-10% gains
- CalibraEval (arXiv:2410.15393): Non-parametric order-preserving debiasing
- Adaptive Repetition (arXiv:2507.17788): Dynamic early-stopping reduces LLM calls by 81%

### 5.2 Multi-Judge Approaches

- SE-Jury (arXiv:2505.20854): 5 evaluation strategies, dynamic team selection
- AgentAuditor (arXiv:2602.09341): Reasoning tree path search outperforms majority vote
- Meta-Judge (arXiv:2505.19477): More bias-resistant than multi-agent debate

### 5.3 Calibration

- Linear Probes (arXiv:2512.22245): Brier score loss on hidden states, 10x computational savings
- RULERS (arXiv:2601.08654): Executable rubrics + Wasserstein calibration
- Bridge (arXiv:2508.12792): Linear transformation model for human-LLM gap

## 6. Key References

| Paper | arXiv ID | Relevance |
|-------|----------|-----------|
| FActScore | 2305.14251 | Foundational atomic fact evaluation |
| SAFE (Google) | 2403.18802 | Multi-step search-augmented verification |
| VITAL | 2510.07083 | Importance-weighted claims -- basis of VitalCorrectnessJudge |
| A Closer Look at Claim Decomposition | 2403.11903 | FActScore sensitivity to decomposition method |
| Decomposition Dilemmas | 2411.02400 | When decomposition hurts vs helps |
| D-FActScore | 2402.05629 | Entity ambiguity causes 10%+ overestimation |
| MiniCheck | 2404.10774 | 770M model = GPT-4 accuracy at 400x lower cost |
| Claimify | 2502.10855 | Best current claim extraction with ambiguity handling |
| DecMetrics | 2509.04483 | Quality metrics for claim decomposition |
| RAGAS | 2309.15217 | Reference-free RAG evaluation framework |
| ARES | 2311.09476 | Automated evaluation with PPI calibration |
| SE-Jury | 2505.20854 | Ensemble judge: 29-140% improvement |
| CalibraEval | 2410.15393 | Distribution calibration for debiasing |
| PINE | 2407.01100 | Position-invariant inference |
| AutoNuggetizer (TREC RAG 2024) | 2504.15068 | Validated automatic nugget evaluation |
| LLMs-as-Judges Survey | 2412.05579 | Comprehensive survey of the field |
| JudgeBench | 2410.12784 | Benchmark: judges near random on hard tasks |
| VeriScore | 2406.19276 | Verifiability classification before checking |
| Trust or Escalate | 2407.18370 | Cascaded evaluation with provable guarantees |
| RULERS | 2601.08654 | Executable rubrics + Wasserstein calibration |
| QuanTemp++ | 2510.22055 | Numerical claim fact-checking benchmark |
| MedScore | 2505.18452 | Domain-adapted claim decomposition |
