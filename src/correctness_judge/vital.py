import asyncio
import logging
from typing import Dict, List, Optional

from litellm import acompletion

from .models import (
    ClaimImportance,
    ClaimVerdict,
    JudgeConfig,
    Nugget,
    VitalScore,
)
from .prompts import (
    CORRECTNESS_JUDGE_MODEL,
    NUGGET_EXTRACTION_PROMPT,
    COT_NUGGET_EXTRACTION_PROMPT,
    CLAIM_DEDUPLICATION_PROMPT,
)
from .base import CorrectnessJudge

logger = logging.getLogger(__name__)


class VitalCorrectnessJudge(CorrectnessJudge):
    def __init__(
        self,
        model: str = CORRECTNESS_JUDGE_MODEL,
        okay_weight: float = 0.5,
        config: Optional[JudgeConfig] = None,
    ):
        super().__init__(model, config)
        self.weights = {
            ClaimImportance.VITAL: 1.0,
            ClaimImportance.OKAY: okay_weight,
            ClaimImportance.NOT_IMPORTANT: 0.0,
        }

    async def evaluate_vital(
        self,
        query: str,
        expected: str,
        actual: str,
    ) -> VitalScore:
        extract = (
            self._extract_nuggets_cot if self.config.use_cot else self._extract_nuggets
        )

        expected_nuggets, actual_nuggets = await asyncio.gather(
            extract(query, expected),
            extract(query, actual),
        )

        if self.config.deduplicate:
            expected_nuggets, actual_nuggets = await asyncio.gather(
                self._deduplicate_nuggets(expected_nuggets),
                self._deduplicate_nuggets(actual_nuggets),
            )

        expected_all_strs = [n.claim for n in expected_nuggets]
        actual_all_strs = [n.claim for n in actual_nuggets]
        expected_weight_map = {n.claim: n for n in expected_nuggets}
        actual_weight_map = {n.claim: n for n in actual_nuggets}

        if not expected_nuggets and not actual_nuggets:
            return VitalScore(
                vital_precision=1.0,
                vital_recall=1.0,
                vital_f1=1.0,
                weighted_precision=1.0,
                weighted_recall=1.0,
                weighted_f1=1.0,
                response_level_precision=True,
                response_level_recall=True,
                expected_nuggets=expected_nuggets,
                actual_nuggets=actual_nuggets,
            )

        if not expected_nuggets:
            return VitalScore(
                vital_precision=0.0,
                vital_recall=1.0,
                vital_f1=0.0,
                weighted_precision=0.0,
                weighted_recall=1.0,
                weighted_f1=0.0,
                response_level_precision=False,
                response_level_recall=True,
                expected_nuggets=expected_nuggets,
                actual_nuggets=actual_nuggets,
            )

        if not actual_nuggets:
            has_vital_expected = any(
                n.importance == ClaimImportance.VITAL for n in expected_nuggets
            )
            return VitalScore(
                vital_precision=1.0,
                vital_recall=0.0,
                vital_f1=0.0,
                weighted_precision=1.0,
                weighted_recall=0.0,
                weighted_f1=0.0,
                response_level_precision=True,
                response_level_recall=not has_vital_expected,
                expected_nuggets=expected_nuggets,
                actual_nuggets=actual_nuggets,
            )

        verify = (
            self._verify_claims_shuffled
            if self.config.shuffle_claims
            else self._verify_claims
        )

        if self.config.shuffle_claims:
            recall_verdicts, precision_verdicts = await asyncio.gather(
                verify(actual, expected_all_strs, seed=self.config.shuffle_seed),
                verify(expected, actual_all_strs, seed=self.config.shuffle_seed),
            )
        else:
            recall_verdicts, precision_verdicts = await asyncio.gather(
                verify(actual, expected_all_strs),
                verify(expected, actual_all_strs),
            )

        for v in recall_verdicts:
            nugget = expected_weight_map.get(v.claim)
            if nugget:
                v.importance = nugget.importance.value
                v.weight = nugget.weight
            else:
                v.importance = ClaimImportance.OKAY.value
                v.weight = self.weights[ClaimImportance.OKAY]

        for v in precision_verdicts:
            nugget = actual_weight_map.get(v.claim)
            if nugget:
                v.importance = nugget.importance.value
                v.weight = nugget.weight
            else:
                v.importance = ClaimImportance.OKAY.value
                v.weight = self.weights[ClaimImportance.OKAY]
            if v.verdict == "not_mentioned":
                v.verdict = "hallucinated"

        vital_recall = [
            v for v in recall_verdicts if v.importance == ClaimImportance.VITAL.value
        ]
        vital_precision = [
            v for v in precision_verdicts if v.importance == ClaimImportance.VITAL.value
        ]

        v_tp = sum(1 for v in vital_recall if v.verdict == "supported")
        v_fn = sum(1 for v in vital_recall if v.verdict != "supported")
        v_fp = sum(
            1 for v in vital_precision if v.verdict in ("contradicted", "hallucinated")
        )
        v_prec = v_tp / (v_tp + v_fp) if (v_tp + v_fp) > 0 else 1.0
        v_rec = v_tp / (v_tp + v_fn) if (v_tp + v_fn) > 0 else 1.0
        v_f1 = (2 * v_prec * v_rec / (v_prec + v_rec)) if (v_prec + v_rec) > 0 else 0.0

        w_tp = sum(v.weight for v in recall_verdicts if v.verdict == "supported")
        w_fn = sum(v.weight for v in recall_verdicts if v.verdict != "supported")
        w_fp = sum(
            v.weight
            for v in precision_verdicts
            if v.verdict in ("contradicted", "hallucinated")
        )
        w_prec = w_tp / (w_tp + w_fp) if (w_tp + w_fp) > 0 else 0.0
        w_rec = w_tp / (w_tp + w_fn) if (w_tp + w_fn) > 0 else 0.0
        w_f1 = (2 * w_prec * w_rec / (w_prec + w_rec)) if (w_prec + w_rec) > 0 else 0.0

        rlp = v_fp == 0
        rlr = v_fn == 0

        return VitalScore(
            vital_precision=round(v_prec, 3),
            vital_recall=round(v_rec, 3),
            vital_f1=round(v_f1, 3),
            weighted_precision=round(w_prec, 3),
            weighted_recall=round(w_rec, 3),
            weighted_f1=round(w_f1, 3),
            response_level_precision=rlp,
            response_level_recall=rlr,
            expected_nuggets=expected_nuggets,
            actual_nuggets=actual_nuggets,
            recall_verdicts=recall_verdicts,
            precision_verdicts=precision_verdicts,
        )

    async def _extract_nuggets(self, query: str, text: str) -> List[Nugget]:
        if len(text.strip()) < 20:
            if text.strip():
                return [
                    Nugget(
                        claim=text.strip(),
                        importance=ClaimImportance.VITAL,
                        weight=self.weights[ClaimImportance.VITAL],
                    )
                ]
            return []

        prompt = NUGGET_EXTRACTION_PROMPT.format(query=query, text=text[:8000])

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
            raw_nuggets = data.get("nuggets", [])

            results = []
            for item in raw_nuggets:
                if isinstance(item, dict):
                    claim_text = item.get("claim", "")
                    importance_str = item.get("importance", "okay").lower()
                    try:
                        importance = ClaimImportance(importance_str)
                    except ValueError:
                        importance = ClaimImportance.OKAY
                    if claim_text:
                        results.append(
                            Nugget(
                                claim=claim_text,
                                importance=importance,
                                weight=self.weights[importance],
                            )
                        )
                elif isinstance(item, str) and item:
                    results.append(
                        Nugget(
                            claim=item,
                            importance=ClaimImportance.OKAY,
                            weight=self.weights[ClaimImportance.OKAY],
                        )
                    )

            return results

        except Exception as e:
            logger.error(f"Nugget extraction failed: {e}")
            return [
                Nugget(
                    claim=text[:500],
                    importance=ClaimImportance.VITAL,
                    weight=self.weights[ClaimImportance.VITAL],
                )
            ]

    async def _extract_nuggets_cot(self, query: str, text: str) -> List[Nugget]:
        """Extract nuggets using chain-of-thought reasoning for better completeness.

        Uses a structured prompt that first identifies query-relevant topics,
        then extracts and rates nuggets per topic. Reduces overlap and improves
        coverage vs single-pass extraction. Also enforces entity disambiguation.

        Based on G-Eval (arXiv:2303.16634, NeurIPS 2023) chain-of-thought
        evaluation and DecMetrics (arXiv:2509.04483) structured decomposition.
        """
        if len(text.strip()) < 20:
            if text.strip():
                return [
                    Nugget(
                        claim=text.strip(),
                        importance=ClaimImportance.VITAL,
                        weight=self.weights[ClaimImportance.VITAL],
                    )
                ]
            return []

        prompt = COT_NUGGET_EXTRACTION_PROMPT.format(query=query, text=text[:8000])

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
            raw_nuggets = data.get("nuggets", [])

            results = []
            for item in raw_nuggets:
                if isinstance(item, dict):
                    claim_text = item.get("claim", "")
                    importance_str = item.get("importance", "okay").lower()
                    try:
                        importance = ClaimImportance(importance_str)
                    except ValueError:
                        importance = ClaimImportance.OKAY
                    if claim_text:
                        results.append(
                            Nugget(
                                claim=claim_text,
                                importance=importance,
                                weight=self.weights[importance],
                            )
                        )

            return results

        except Exception as e:
            logger.error(f"CoT nugget extraction failed, falling back to basic: {e}")
            return await self._extract_nuggets(query, text)

    async def _deduplicate_nuggets(self, nuggets: List[Nugget]) -> List[Nugget]:
        """Remove semantically overlapping nuggets via LLM-based merging.

        Automatic nugget extraction produces 15-30% overlapping claims that
        inflate precision/recall scores. This merges redundant nuggets while
        preserving the highest importance label from each group using
        source_indices from the LLM response for robust mapping.

        Based on:
        - TREC 2024 RAG Track AutoNuggetizer (arXiv:2504.15068)
        - GINGER nugget clustering (arXiv:2503.18174)
        """
        if len(nuggets) <= 1:
            return nuggets

        claims = [n.claim for n in nuggets]

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
                return nuggets

            importance_priority = {
                ClaimImportance.VITAL: 0,
                ClaimImportance.OKAY: 1,
                ClaimImportance.NOT_IMPORTANT: 2,
            }

            results = []
            for item in deduped:
                if isinstance(item, dict):
                    canonical = str(item.get("claim", ""))
                    source_indices = item.get("source_indices", [])
                elif isinstance(item, str):
                    canonical = item
                    source_indices = []
                else:
                    continue

                if not canonical:
                    continue

                best_importance = ClaimImportance.NOT_IMPORTANT

                if source_indices:
                    for idx in source_indices:
                        zero_idx = int(idx) - 1
                        if 0 <= zero_idx < len(nuggets):
                            orig_importance = nuggets[zero_idx].importance
                            if importance_priority.get(
                                orig_importance, 2
                            ) < importance_priority.get(best_importance, 2):
                                best_importance = orig_importance

                if best_importance == ClaimImportance.NOT_IMPORTANT:
                    for n in nuggets:
                        if n.claim in canonical or canonical in n.claim:
                            if importance_priority.get(
                                n.importance, 2
                            ) < importance_priority.get(best_importance, 2):
                                best_importance = n.importance

                if best_importance == ClaimImportance.NOT_IMPORTANT:
                    best_importance = ClaimImportance.OKAY

                results.append(
                    Nugget(
                        claim=canonical,
                        importance=best_importance,
                        weight=self.weights[best_importance],
                    )
                )

            return results if results else nuggets

        except Exception as e:
            logger.error(f"Nugget deduplication failed, keeping originals: {e}")
            return nuggets
