from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class CorrectnessVerdict(str, Enum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"


@dataclass
class CorrectnessScore:
    verdict: CorrectnessVerdict
    confidence: float
    differences: List[str] = field(default_factory=list)
    explanation: str = ""

    @property
    def is_correct(self) -> bool:
        return self.verdict == CorrectnessVerdict.CORRECT

    @property
    def is_passing(self) -> bool:
        return self.confidence >= 0.9 and self.verdict in (
            CorrectnessVerdict.CORRECT,
            CorrectnessVerdict.PARTIALLY_CORRECT,
        )


class ClaimImportance(str, Enum):
    VITAL = "vital"
    OKAY = "okay"
    NOT_IMPORTANT = "not_important"


IMPORTANCE_WEIGHTS = {
    ClaimImportance.VITAL: 1.0,
    ClaimImportance.OKAY: 0.5,
    ClaimImportance.NOT_IMPORTANT: 0.0,
}


@dataclass
class ClaimVerdict:
    claim: str
    verdict: str
    evidence: str = ""
    importance: Optional[str] = None
    weight: float = 1.0
    probability: float = 1.0


@dataclass
class Nugget:
    claim: str
    importance: ClaimImportance
    weight: float = 1.0


@dataclass
class JudgeConfig:
    use_cot: bool = False
    deduplicate: bool = False
    shuffle_claims: bool = False
    shuffle_seed: int | None = None
    numerical_tolerance: float = 0.05


@dataclass
class DecomposedCorrectnessScore:
    precision: float
    recall: float
    f1: float
    expected_claims: List[str] = field(default_factory=list)
    actual_claims: List[str] = field(default_factory=list)
    supported: List[ClaimVerdict] = field(default_factory=list)
    contradicted: List[ClaimVerdict] = field(default_factory=list)
    not_mentioned: List[ClaimVerdict] = field(default_factory=list)
    hallucinated: List[ClaimVerdict] = field(default_factory=list)
    soft_precision: Optional[float] = None
    soft_recall: Optional[float] = None
    soft_f1: Optional[float] = None

    @property
    def verdict(self) -> CorrectnessVerdict:
        if self.f1 >= 0.9 and len(self.contradicted) == 0:
            return CorrectnessVerdict.CORRECT
        if self.f1 <= 0.3 or len(self.contradicted) > len(self.supported):
            return CorrectnessVerdict.INCORRECT
        return CorrectnessVerdict.PARTIALLY_CORRECT

    @property
    def is_passing(self) -> bool:
        return self.f1 >= 0.8

    @property
    def primary_f1(self) -> float:
        return self.f1

    def to_correctness_score(self) -> "CorrectnessScore":
        differences = []
        for c in self.contradicted:
            differences.append(f"CONTRADICTED: {c.claim} (evidence: {c.evidence})")
        for c in self.not_mentioned:
            differences.append(f"MISSING: {c.claim}")
        for c in self.hallucinated:
            differences.append(f"HALLUCINATED: {c.claim} (evidence: {c.evidence})")

        parts = [f"P={self.precision:.2f} R={self.recall:.2f} F1={self.f1:.2f}"]
        parts.append(f"{len(self.supported)} supported")
        if self.contradicted:
            parts.append(f"{len(self.contradicted)} contradicted")
        if self.not_mentioned:
            parts.append(f"{len(self.not_mentioned)} missing")
        if self.hallucinated:
            parts.append(f"{len(self.hallucinated)} hallucinated")

        return CorrectnessScore(
            verdict=self.verdict,
            confidence=self.primary_f1,
            differences=differences,
            explanation=", ".join(parts),
        )


@dataclass
class VitalScore:
    vital_precision: float
    vital_recall: float
    vital_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    response_level_precision: bool
    response_level_recall: bool
    expected_nuggets: List[Nugget] = field(default_factory=list)
    actual_nuggets: List[Nugget] = field(default_factory=list)
    recall_verdicts: List[ClaimVerdict] = field(default_factory=list)
    precision_verdicts: List[ClaimVerdict] = field(default_factory=list)

    @property
    def verdict(self) -> CorrectnessVerdict:
        if not self.response_level_precision or not self.response_level_recall:
            return CorrectnessVerdict.INCORRECT
        if self.vital_f1 >= 0.9:
            return CorrectnessVerdict.CORRECT
        if self.vital_f1 <= 0.3:
            return CorrectnessVerdict.INCORRECT
        return CorrectnessVerdict.PARTIALLY_CORRECT

    @property
    def is_passing(self) -> bool:
        return self.response_level_precision and self.response_level_recall

    def to_correctness_score(self) -> CorrectnessScore:
        differences = []
        for c in self.recall_verdicts:
            if (
                c.verdict == "contradicted"
                and c.importance == ClaimImportance.VITAL.value
            ):
                differences.append(
                    f"VITAL CONTRADICTED: {c.claim} (evidence: {c.evidence})"
                )
            elif (
                c.verdict == "not_mentioned"
                and c.importance == ClaimImportance.VITAL.value
            ):
                differences.append(f"VITAL MISSING: {c.claim}")
        for c in self.precision_verdicts:
            if c.verdict in ("contradicted", "hallucinated"):
                if c.importance == ClaimImportance.VITAL.value:
                    differences.append(
                        f"VITAL HALLUCINATED: {c.claim} (evidence: {c.evidence})"
                    )

        parts = [
            f"VitalP={self.vital_precision:.2f}",
            f"VitalR={self.vital_recall:.2f}",
            f"VitalF1={self.vital_f1:.2f}",
            f"wP={self.weighted_precision:.2f}",
            f"wR={self.weighted_recall:.2f}",
            f"wF1={self.weighted_f1:.2f}",
            f"RLP={'pass' if self.response_level_precision else 'FAIL'}",
            f"RLR={'pass' if self.response_level_recall else 'FAIL'}",
        ]

        return CorrectnessScore(
            verdict=self.verdict,
            confidence=self.vital_f1,
            differences=differences,
            explanation=", ".join(parts),
        )
