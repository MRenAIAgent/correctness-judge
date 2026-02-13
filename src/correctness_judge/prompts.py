CORRECTNESS_JUDGE_MODEL = "anthropic/claude-sonnet-4-20250514"


CLAIM_EXTRACTION_PROMPT = """Extract every distinct factual claim from the following text. A claim is a single atomic statement that can be independently verified as true or false.

Rules:
- Each claim should be self-contained (understandable without the other claims).
- Split compound statements into separate claims.
- Ignore opinions, hedging language, and filler text.
- Include numerical facts, named entities, relationships, dates, and causal statements.
- Do NOT include formatting, structure, or stylistic observations.
- Each claim must explicitly name the entity it refers to (no pronouns or ambiguous references).

TEXT:
{text}

Respond with JSON only:
{{
  "claims": ["claim 1", "claim 2", ...]
}}"""


CLAIM_VERIFICATION_PROMPT = """You are verifying whether specific factual claims are supported by a reference text.

For EACH claim below, determine if the reference text supports it, contradicts it, or neither (the reference is silent on it). Also estimate the probability (0.0 to 1.0) that the claim is true given the reference.

REFERENCE TEXT (ground truth):
{reference}

CLAIMS TO VERIFY:
{claims}

For each claim, respond with one of:
- "supported": The reference text contains information that confirms this claim.
- "contradicted": The reference text contains information that directly contradicts this claim.
- "not_mentioned": The reference text does not address this claim at all.

For probability:
- A clearly supported claim should have probability close to 1.0.
- A clearly contradicted claim should have probability close to 0.0.
- A partially supported claim (correct direction but wrong details) might be 0.3-0.7.
- A claim not mentioned in the reference should have probability 0.5 (unknown).

Respond with JSON only:
{{
  "verdicts": [
    {{"claim": "the claim text", "verdict": "supported|contradicted|not_mentioned", "probability": 0.0-1.0, "evidence": "brief quote or explanation from reference"}}
  ]
}}"""


CORRECTNESS_PROMPT = """You are a strict factual correctness evaluator. Your job is to find every difference between an expected answer and an actual answer.

IMPORTANT: Look for differences first. Do NOT confirm matches - hunt for mismatches.

QUERY:
{query}

EXPECTED ANSWER (ground truth):
{expected}

ACTUAL ANSWER (to evaluate):
{actual}

INSTRUCTIONS:
1. List every factual difference between the expected and actual answers.
2. A difference is any piece of information that is present in one but not the other, or that contradicts between them.
3. Minor phrasing differences that preserve meaning are NOT differences.
4. If the actual answer contains extra correct information beyond what is expected, that is NOT a difference.
5. If there are zero differences, the answer is correct.

Respond with JSON only:
{{
  "verdict": "correct" | "partially_correct" | "incorrect",
  "confidence": <0.0 to 1.0>,
  "differences": ["list each specific factual mismatch or missing element"],
  "explanation": "one sentence summary"
}}

Rules for verdict:
- "correct": Zero meaningful differences. Confidence should be >= 0.9.
- "partially_correct": Some facts match but others are missing or wrong. Confidence reflects the proportion correct.
- "incorrect": Core answer is wrong or contradicts the expected answer. Confidence should be <= 0.3."""


CORRECTNESS_PROMPT_STRUCTURED = """You are a strict factual correctness evaluator comparing structured data.

IMPORTANT: Check each field independently. Report every field-level mismatch.

QUERY:
{query}

EXPECTED OUTPUT (ground truth):
{expected}

ACTUAL OUTPUT (to evaluate):
{actual}

INSTRUCTIONS:
1. Compare each field in the expected output against the actual output.
2. For each field, determine if the actual value matches the expected value.
3. String comparisons should be case-insensitive and ignore minor formatting.
4. Numeric comparisons should allow small tolerance (e.g., rounding differences).
5. List fields are matched if they contain the same elements regardless of order.
6. Missing fields in the actual output count as differences.
7. Extra fields in the actual output that are not in expected do NOT count as differences.

Respond with JSON only:
{{
  "verdict": "correct" | "partially_correct" | "incorrect",
  "confidence": <0.0 to 1.0>,
  "differences": ["field_name: expected X, got Y"],
  "explanation": "one sentence summary"
}}"""


CLAIM_DEDUPLICATION_PROMPT = """You are given a list of factual claims extracted from the same text. Some claims may overlap or be redundant -- one claim may be a subset of another, or two claims may express the same fact differently.

Identify groups of semantically overlapping claims and merge each group into a single canonical claim that preserves all information. Keep claims that are truly independent.

CLAIMS:
{claims}

Rules:
- If claim A strictly entails claim B (A contains all info of B plus more), keep only A.
- If two claims express the same fact with different wording, merge into one.
- If claims are truly independent (different facts), keep both.
- Preserve numerical precision from the more specific claim when merging.
- For each output claim, list which input claim numbers were merged into it.

Respond with JSON only:
{{
  "deduplicated_claims": [
    {{"claim": "canonical claim text", "source_indices": [1, 3]}},
    {{"claim": "another independent claim", "source_indices": [2]}}
  ]
}}"""


COT_CLAIM_EXTRACTION_PROMPT = """Extract every distinct factual claim from the following text using a structured approach.

Step 1: Identify the main topics or themes in the text.
Step 2: For each topic, extract individual atomic factual claims.
Step 3: Ensure each claim is self-contained and independently verifiable.
Step 4: Remove any duplicates or claims that are subsumed by more specific ones.

A claim is a single atomic statement that can be independently verified as true or false.

Rules:
- Each claim should be self-contained (understandable without the other claims).
- Split compound statements into separate claims.
- Ignore opinions, hedging language, and filler text.
- Include numerical facts, named entities, relationships, dates, and causal statements.
- Do NOT include formatting, structure, or stylistic observations.
- Each claim must explicitly name the entity it refers to (no pronouns or ambiguous references).

TEXT:
{text}

Respond with JSON only:
{{
  "topics": ["topic 1", "topic 2", ...],
  "claims": ["claim 1", "claim 2", ...]
}}"""


COT_NUGGET_EXTRACTION_PROMPT = """Extract every distinct factual claim from the text below as nuggets using a structured approach, and rate each nugget's importance for answering the query.

Step 1: Identify the main topics or themes relevant to the query.
Step 2: For each topic, extract individual atomic factual claims.
Step 3: Rate each claim's importance for answering the query.
Step 4: Remove any duplicates or claims subsumed by more specific ones.

A nugget is a minimal, atomic claim that is a correct and useful piece of information in answering the query.

QUERY (what the user asked):
{query}

TEXT:
{text}

Rules for extraction:
- Each nugget should be a single atomic fact, self-contained and independently verifiable.
- Split compound statements into separate nuggets.
- Ignore opinions, hedging, and filler text.
- Each nugget must explicitly name the entity it refers to (no pronouns or ambiguous references).

Rules for importance rating:
- "vital": This nugget is essential to answer the query. A response missing this nugget is incomplete.
- "okay": This nugget is useful supporting information but not strictly necessary.
- "not_important": This nugget is background context or a tangential detail.

Respond with JSON only:
{{
  "topics": ["topic 1", "topic 2", ...],
  "nuggets": [
    {{"claim": "the factual statement", "importance": "vital|okay|not_important"}}
  ]
}}"""


NUMERICAL_COMPARISON_PROMPT = """Compare these two numerical or date values and determine if they represent the same quantity within reasonable tolerance.

EXPECTED VALUE: {expected}
ACTUAL VALUE: {actual}

Rules:
- Numbers: Allow tolerance for rounding (e.g., 3.2B vs 3.19B are the same).
- Percentages: Allow +/- 1% tolerance (e.g., 45% vs 44.7% are the same).
- Dates: Exact match required for specific dates. Approximate match for ranges.
- Currency: Same amount in same currency required. Rounding differences OK.
- Units: Must be compatible (e.g., 1000m and 1km are the same).

Respond with JSON only:
{{
  "match": true or false,
  "reason": "brief explanation"
}}"""


NUGGET_EXTRACTION_PROMPT = """Extract every distinct factual claim from the text below as nuggets, and rate each nugget's importance for answering the query.

A nugget is a minimal, atomic claim that is a correct and useful piece of information in answering the query.

QUERY (what the user asked):
{query}

TEXT:
{text}

Rules for extraction:
- Each nugget should be a single atomic fact, self-contained and independently verifiable.
- Split compound statements into separate nuggets.
- Ignore opinions, hedging, and filler text.
- Each nugget must explicitly name the entity it refers to (no pronouns or ambiguous references).

Rules for importance rating:
- "vital": This nugget is essential to answer the query. A response missing this nugget is incomplete.
- "okay": This nugget is useful supporting information but not strictly necessary.
- "not_important": This nugget is background context or a tangential detail.

Respond with JSON only:
{{
  "nuggets": [
    {{"claim": "the factual statement", "importance": "vital|okay|not_important"}}
  ]
}}"""
