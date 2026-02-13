from correctness_judge.base import CorrectnessJudge
from correctness_judge.models import CorrectnessScore, CorrectnessVerdict


class TestAggregateScores:
    def test_empty(self):
        result = CorrectnessJudge.aggregate_scores([])
        assert result["total"] == 0
        assert result["accuracy"] == 0.0

    def test_all_correct(self):
        scores = [
            CorrectnessScore(verdict=CorrectnessVerdict.CORRECT, confidence=0.95),
            CorrectnessScore(verdict=CorrectnessVerdict.CORRECT, confidence=0.99),
        ]
        result = CorrectnessJudge.aggregate_scores(scores)
        assert result["total"] == 2
        assert result["correct"] == 2
        assert result["accuracy"] == 1.0

    def test_mixed(self):
        scores = [
            CorrectnessScore(verdict=CorrectnessVerdict.CORRECT, confidence=0.95),
            CorrectnessScore(verdict=CorrectnessVerdict.PARTIALLY_CORRECT, confidence=0.7),
            CorrectnessScore(verdict=CorrectnessVerdict.INCORRECT, confidence=0.1),
        ]
        result = CorrectnessJudge.aggregate_scores(scores)
        assert result["total"] == 3
        assert result["correct"] == 1
        assert result["partial"] == 1
        assert result["incorrect"] == 1
        assert abs(result["accuracy"] - 1 / 3) < 0.01

    def test_pass_rate(self):
        scores = [
            CorrectnessScore(verdict=CorrectnessVerdict.CORRECT, confidence=0.95),
            CorrectnessScore(verdict=CorrectnessVerdict.CORRECT, confidence=0.5),
        ]
        result = CorrectnessJudge.aggregate_scores(scores)
        assert result["pass_rate"] == 0.5
