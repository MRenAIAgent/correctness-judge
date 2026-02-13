import asyncio
import json
import logging
import random
import re
from typing import Dict, Any, List, Optional, Union

from litellm import acompletion

from .models import (
    CorrectnessVerdict,
    CorrectnessScore,
    ClaimVerdict,
    DecomposedCorrectnessScore,
    JudgeConfig,
)
from .prompts import (
    CORRECTNESS_JUDGE_MODEL,
    CLAIM_EXTRACTION_PROMPT,
    CLAIM_VERIFICATION_PROMPT,
    CORRECTNESS_PROMPT,
    CORRECTNESS_PROMPT_STRUCTURED,
    COT_CLAIM_EXTRACTION_PROMPT,
    CLAIM_DEDUPLICATION_PROMPT,
    NUMERICAL_COMPARISON_PROMPT,
)

logger = logging.getLogger(__name__)


class CorrectnessJudge:
    def __init__(
        self,
        model: str = CORRECTNESS_JUDGE_MODEL,
        config: Optional[JudgeConfig] = None,
    ):
        self.model = model
        self.config = config or JudgeConfig()

    async def evaluate(
        self,
        query: str,
        expected: Union[str, Dict[str, Any], List[Any]],
        actual: Union[str, Dict[str, Any], List[Any]],
    ) -> CorrectnessScore:
        is_structured = isinstance(expected, (dict, list)) or isinstance(
            actual, (dict, list)
        )

        if is_structured:
            expected_str = (
                json.dumps(expected, indent=2)
                if not isinstance(expected, str)
                else expected
            )
            actual_str = (
                json.dumps(actual, indent=2) if not isinstance(actual, str) else actual
            )
            prompt = CORRECTNESS_PROMPT_STRUCTURED.format(
                query=query, expected=expected_str, actual=actual_str
            )
        else:
            prompt = CORRECTNESS_PROMPT.format(
                query=query, expected=str(expected), actual=str(actual)
            )

        try:
            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
                response_format={"type": "json_object"},
            )

            response_text = response.choices[0].message.content.strip()
            parsed = self._parse_response(response_text)
            return parsed

        except Exception as e:
            logger.error(f"Correctness judge error: {e}")
            return CorrectnessScore(
                verdict=CorrectnessVerdict.INCORRECT,
                confidence=0.0,
                differences=[f"Judge error: {str(e)}"],
                explanation=f"Evaluation failed: {str(e)}",
            )

    async def evaluate_batch(
        self,
        items: List[Dict[str, Any]],
        max_concurrent: int = 10,
    ) -> List[CorrectnessScore]:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_evaluate(item):
            async with semaphore:
                return await self.evaluate(
                    query=item["query"],
                    expected=item["expected"],
                    actual=item["actual"],
                )

        tasks = [bounded_evaluate(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scores = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch item {i} failed: {result}")
                scores.append(
                    CorrectnessScore(
                        verdict=CorrectnessVerdict.INCORRECT,
                        confidence=0.0,
                        differences=[f"Error: {str(result)}"],
                        explanation=f"Evaluation failed: {str(result)}",
                    )
                )
            else:
                scores.append(result)

        return scores

    async def evaluate_long_form(
        self,
        query: str,
        expected: str,
        actual: str,
    ) -> DecomposedCorrectnessScore:
        extract = (
            self._extract_claims_cot if self.config.use_cot else self._extract_claims
        )

        expected_claims, actual_claims = await asyncio.gather(
            extract(expected),
            extract(actual),
        )

        if self.config.deduplicate:
            expected_claims, actual_claims = await asyncio.gather(
                self._deduplicate_claims(expected_claims),
                self._deduplicate_claims(actual_claims),
            )

        if not expected_claims and not actual_claims:
            return DecomposedCorrectnessScore(
                precision=1.0,
                recall=1.0,
                f1=1.0,
                expected_claims=[],
                actual_claims=[],
            )

        if not expected_claims:
            return DecomposedCorrectnessScore(
                precision=0.0,
                recall=1.0,
                f1=0.0,
                expected_claims=[],
                actual_claims=actual_claims,
                hallucinated=[
                    ClaimVerdict(claim=c, verdict="hallucinated") for c in actual_claims
                ],
            )

        if not actual_claims:
            return DecomposedCorrectnessScore(
                precision=1.0,
                recall=0.0,
                f1=0.0,
                expected_claims=expected_claims,
                actual_claims=[],
                not_mentioned=[
                    ClaimVerdict(claim=c, verdict="not_mentioned")
                    for c in expected_claims
                ],
            )

        verify = (
            self._verify_claims_shuffled
            if self.config.shuffle_claims
            else self._verify_claims
        )

        if self.config.shuffle_claims:
            recall_verdicts, precision_verdicts = await asyncio.gather(
                verify(actual, expected_claims, seed=self.config.shuffle_seed),
                verify(expected, actual_claims, seed=self.config.shuffle_seed),
            )
        else:
            recall_verdicts, precision_verdicts = await asyncio.gather(
                verify(actual, expected_claims),
                verify(expected, actual_claims),
            )

        supported = []
        contradicted = []
        not_mentioned = []
        for v in recall_verdicts:
            if v.verdict == "supported":
                supported.append(v)
            elif v.verdict == "contradicted":
                contradicted.append(v)
            else:
                not_mentioned.append(v)

        hallucinated = []
        for v in precision_verdicts:
            if v.verdict == "contradicted":
                hallucinated.append(v)
            elif v.verdict == "not_mentioned":
                hallucinated.append(
                    ClaimVerdict(
                        claim=v.claim,
                        verdict="hallucinated",
                        evidence=v.evidence,
                        probability=v.probability,
                    )
                )

        tp = len(supported)
        fn = len(not_mentioned) + len(contradicted)
        fp = len(hallucinated)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        soft_recall_sum = sum(v.probability for v in recall_verdicts)
        soft_recall_total = len(recall_verdicts)
        soft_precision_sum = sum(v.probability for v in precision_verdicts)
        soft_precision_total = len(precision_verdicts)

        s_recall = soft_recall_sum / soft_recall_total if soft_recall_total > 0 else 0.0
        s_precision = (
            soft_precision_sum / soft_precision_total
            if soft_precision_total > 0
            else 0.0
        )
        s_f1 = (
            (2 * s_precision * s_recall / (s_precision + s_recall))
            if (s_precision + s_recall) > 0
            else 0.0
        )

        return DecomposedCorrectnessScore(
            precision=round(precision, 3),
            recall=round(recall, 3),
            f1=round(f1, 3),
            expected_claims=expected_claims,
            actual_claims=actual_claims,
            supported=supported,
            contradicted=contradicted,
            not_mentioned=not_mentioned,
            hallucinated=hallucinated,
            soft_precision=round(s_precision, 3),
            soft_recall=round(s_recall, 3),
            soft_f1=round(s_f1, 3),
        )

    async def _extract_claims(self, text: str) -> List[str]:
        if len(text.strip()) < 20:
            return [text.strip()] if text.strip() else []

        prompt = CLAIM_EXTRACTION_PROMPT.format(text=text[:8000])

        try:
            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            response_text = response.choices[0].message.content.strip()
            data = self._parse_json(response_text)
            claims = data.get("claims", [])
            if not isinstance(claims, list):
                return [str(claims)]
            return [str(c) for c in claims if c]
        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            return [text[:500]]

    async def _extract_claims_cot(self, text: str) -> List[str]:
        """Extract claims using chain-of-thought reasoning for better completeness.

        Uses a structured two-stage prompt that first identifies topics/themes,
        then extracts claims per topic. This reduces claim overlap and improves
        coverage compared to direct extraction.

        Also enforces entity disambiguation by requiring explicit entity naming
        in each claim (no pronouns or ambiguous references).

        Based on G-Eval (arXiv:2303.16634, NeurIPS 2023) chain-of-thought
        evaluation and DecMetrics (arXiv:2509.04483) structured claim
        decomposition.
        """
        if len(text.strip()) < 20:
            return [text.strip()] if text.strip() else []

        prompt = COT_CLAIM_EXTRACTION_PROMPT.format(text=text[:8000])

        try:
            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=3000,
                response_format={"type": "json_object"},
            )
            response_text = response.choices[0].message.content.strip()
            data = self._parse_json(response_text)
            claims = data.get("claims", [])
            if not isinstance(claims, list):
                return [str(claims)]
            return [str(c) for c in claims if c]
        except Exception as e:
            logger.error(f"CoT claim extraction failed, falling back to basic: {e}")
            return await self._extract_claims(text)

    async def _deduplicate_claims(self, claims: List[str]) -> List[str]:
        """Remove semantically overlapping or redundant claims via LLM merging.

        Automatic claim extraction consistently produces overlapping claims
        (e.g., "GDP grew in Q3" and "GDP growth was 3% in Q3"). Counting both
        inflates precision/recall by 15-30%.

        This method identifies groups of semantically overlapping claims and
        merges each group into a single canonical claim preserving all info.

        Based on findings from:
        - TREC 2024 RAG Track AutoNuggetizer (arXiv:2504.15068)
        - GINGER nugget clustering (arXiv:2503.18174)
        """
        if len(claims) <= 1:
            return claims

        claims_text = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
        prompt = CLAIM_DEDUPLICATION_PROMPT.format(claims=claims_text)

        try:
            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            response_text = response.choices[0].message.content.strip()
            data = self._parse_json(response_text)
            deduped = data.get("deduplicated_claims", [])
            if not isinstance(deduped, list) or not deduped:
                return claims
            result = []
            for item in deduped:
                if isinstance(item, dict):
                    claim_text = item.get("claim", "")
                    if claim_text:
                        result.append(str(claim_text))
                elif isinstance(item, str) and item:
                    result.append(item)
            return result if result else claims
        except Exception as e:
            logger.error(f"Claim deduplication failed, keeping originals: {e}")
            return claims

    async def _compare_numerical_claims(
        self, expected_value: str, actual_value: str
    ) -> Dict[str, Any]:
        """Compare numerical or date values with appropriate tolerance.

        LLMs handle numerical comparisons inconsistently -- "approximately
        $3.2 billion" vs "$3.19 billion" may be judged differently across
        runs. This method provides deterministic comparison with configurable
        tolerance for numbers, percentages, dates, and currency.

        First attempts programmatic comparison via regex-based number
        extraction. Falls back to LLM comparison for complex cases
        (unit conversions, date ranges, etc.).

        Based on QuanTemp++ (arXiv:2510.22055) numerical claim fact-checking.
        """
        expected_num = self._try_parse_number(expected_value)
        actual_num = self._try_parse_number(actual_value)

        if expected_num is not None and actual_num is not None:
            if expected_num == 0 and actual_num == 0:
                return {"match": True, "reason": "both zero"}
            if expected_num == 0:
                return {
                    "match": abs(actual_num) < 1e-9,
                    "reason": f"expected 0, got {actual_num}",
                }
            relative_diff = abs(expected_num - actual_num) / max(
                abs(expected_num), 1e-9
            )
            tolerance = self.config.numerical_tolerance
            is_match = relative_diff < tolerance
            pct = int(tolerance * 100)
            return {
                "match": is_match,
                "reason": (
                    f"relative diff {relative_diff:.4f} "
                    f"({'within' if is_match else 'exceeds'} {pct}% tolerance)"
                ),
            }

        try:
            prompt = NUMERICAL_COMPARISON_PROMPT.format(
                expected=expected_value, actual=actual_value
            )
            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            response_text = response.choices[0].message.content.strip()
            data = self._parse_json(response_text)
            return {
                "match": bool(data.get("match", False)),
                "reason": data.get("reason", ""),
            }
        except Exception as e:
            logger.error(f"Numerical comparison failed: {e}")
            return {"match": False, "reason": f"comparison error: {e}"}

    @staticmethod
    def _try_parse_number(text: str) -> float | None:
        """Extract a numeric value from text, handling common formats.

        Supports: plain numbers, percentages, currency symbols, suffixes
        (K, M, B, T), comma-separated numbers, and spaced word forms
        (e.g. "1.5 million", "3 billion").
        """
        text = text.strip()

        text = re.sub(r"^[$\u20ac\u00a3\u00a5]", "", text)
        text = text.replace(",", "")
        text = text.rstrip("%")

        multipliers = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}

        word_match = re.match(
            r"^([+-]?\d+\.?\d*)\s+(thousand|million|billion|trillion)$",
            text,
            re.I,
        )
        if word_match:
            base = float(word_match.group(1))
            word = word_match.group(2).lower()
            word_map = {
                "thousand": 1e3,
                "million": 1e6,
                "billion": 1e9,
                "trillion": 1e12,
            }
            return base * word_map[word]

        suffix_match = re.match(r"^([+-]?\d+\.?\d*)\s*([kmbt])(?:illion)?$", text, re.I)
        if suffix_match:
            base = float(suffix_match.group(1))
            suffix = suffix_match.group(2).lower()
            return base * multipliers.get(suffix, 1)

        try:
            return float(text)
        except (ValueError, TypeError):
            return None

    async def _verify_claims(
        self, reference: str, claims: List[str]
    ) -> List[ClaimVerdict]:
        if not claims:
            return []

        claims_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
        prompt = CLAIM_VERIFICATION_PROMPT.format(
            reference=reference[:8000], claims=claims_text
        )

        try:
            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            response_text = response.choices[0].message.content.strip()
            data = self._parse_json(response_text)
            verdicts_raw = data.get("verdicts", [])

            verdicts = []
            for v in verdicts_raw:
                if isinstance(v, dict):
                    raw_prob = v.get("probability")
                    if raw_prob is not None:
                        prob = max(0.0, min(1.0, float(raw_prob)))
                    else:
                        verdict_str = v.get("verdict", "not_mentioned")
                        prob = (
                            1.0
                            if verdict_str == "supported"
                            else 0.0 if verdict_str == "contradicted" else 0.5
                        )
                    verdicts.append(
                        ClaimVerdict(
                            claim=v.get("claim", ""),
                            verdict=v.get("verdict", "not_mentioned"),
                            evidence=v.get("evidence", ""),
                            probability=prob,
                        )
                    )

            if len(verdicts) < len(claims):
                logger.warning(
                    f"LLM returned {len(verdicts)} verdicts for {len(claims)} claims, "
                    f"backfilling {len(claims) - len(verdicts)} as not_mentioned"
                )
                verified_claims = {v.claim for v in verdicts}
                for c in claims:
                    if c not in verified_claims and not any(
                        c in v.claim for v in verdicts
                    ):
                        verdicts.append(
                            ClaimVerdict(
                                claim=c,
                                verdict="not_mentioned",
                                evidence="claim not returned by verifier",
                                probability=0.5,
                            )
                        )

            return verdicts
        except Exception as e:
            logger.error(f"Claim verification failed: {e}")
            return [
                ClaimVerdict(
                    claim=c,
                    verdict="not_mentioned",
                    evidence=f"Error: {e}",
                    probability=0.5,
                )
                for c in claims
            ]

    async def _verify_claims_shuffled(
        self, reference: str, claims: List[str], seed: int | None = None
    ) -> List[ClaimVerdict]:
        """Verify claims with randomized ordering to mitigate position bias.

        LLMs exhibit primacy/recency bias in batch verification -- claims
        near the top of a numbered list receive systematically different
        treatment than those in the middle or bottom. Shuffling the claim
        order before sending to the LLM and then restoring the original
        order eliminates this systematic bias.

        Pass a fixed seed for reproducible results across runs.

        Based on:
        - Position bias in LLM-as-judge (arXiv:2406.07791)
        - PINE: position-invariant evaluation (arXiv:2407.01100)
        """
        if not claims:
            return []

        rng = random.Random(seed)
        indices = list(range(len(claims)))
        rng.shuffle(indices)
        shuffled_claims = [claims[i] for i in indices]

        shuffled_verdicts = await self._verify_claims(reference, shuffled_claims)

        inverse = [0] * len(indices)
        for new_pos, old_pos in enumerate(indices):
            inverse[old_pos] = new_pos

        restored = []
        for orig_idx in range(len(claims)):
            shuffled_idx = inverse[orig_idx]
            if shuffled_idx < len(shuffled_verdicts):
                restored.append(shuffled_verdicts[shuffled_idx])
            else:
                restored.append(
                    ClaimVerdict(
                        claim=claims[orig_idx],
                        verdict="not_mentioned",
                        evidence="",
                        probability=0.5,
                    )
                )

        return restored

    def _parse_json(self, text: str) -> Dict[str, Any]:
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                return json.loads(match.group(0))
            return {}

    def _parse_response(self, text: str) -> CorrectnessScore:
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        try:
            data = json.loads(text.strip())
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                data = json.loads(match.group(0))
            else:
                return CorrectnessScore(
                    verdict=CorrectnessVerdict.INCORRECT,
                    confidence=0.0,
                    differences=["Could not parse judge response"],
                    explanation=f"Parse error: {text[:200]}",
                )

        verdict_str = data.get("verdict", "incorrect").lower()
        try:
            verdict = CorrectnessVerdict(verdict_str)
        except ValueError:
            verdict = CorrectnessVerdict.INCORRECT

        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        differences = data.get("differences", [])
        if not isinstance(differences, list):
            differences = [str(differences)]

        if verdict == CorrectnessVerdict.CORRECT and len(differences) > 0:
            logger.warning(
                f"Judge said correct but listed {len(differences)} differences, downgrading"
            )
            verdict = CorrectnessVerdict.PARTIALLY_CORRECT
            confidence = min(confidence, 0.7)

        return CorrectnessScore(
            verdict=verdict,
            confidence=confidence,
            differences=differences,
            explanation=data.get("explanation", ""),
        )

    @staticmethod
    def aggregate_scores(scores: List[CorrectnessScore]) -> Dict[str, Any]:
        if not scores:
            return {
                "total": 0,
                "correct": 0,
                "partial": 0,
                "incorrect": 0,
                "accuracy": 0.0,
            }

        correct = sum(1 for s in scores if s.verdict == CorrectnessVerdict.CORRECT)
        partial = sum(
            1 for s in scores if s.verdict == CorrectnessVerdict.PARTIALLY_CORRECT
        )
        incorrect = sum(1 for s in scores if s.verdict == CorrectnessVerdict.INCORRECT)
        avg_confidence = sum(s.confidence for s in scores) / len(scores)
        pass_rate = sum(1 for s in scores if s.is_passing) / len(scores)

        return {
            "total": len(scores),
            "correct": correct,
            "partial": partial,
            "incorrect": incorrect,
            "accuracy": correct / len(scores),
            "pass_rate": pass_rate,
            "avg_confidence": round(avg_confidence, 3),
        }
