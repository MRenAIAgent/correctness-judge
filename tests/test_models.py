from correctness_judge.models import (
    CorrectnessVerdict,
    CorrectnessScore,
    ClaimImportance,
    IMPORTANCE_WEIGHTS,
    ClaimVerdict,
    Nugget,
    JudgeConfig,
    DecomposedCorrectnessScore,
    VitalScore,
)


class TestCorrectnessScore:
    def test_is_correct(self):
        score = CorrectnessScore(verdict=CorrectnessVerdict.CORRECT, confidence=0.95)
        assert score.is_correct is True

    def test_is_not_correct(self):
        score = CorrectnessScore(verdict=CorrectnessVerdict.INCORRECT, confidence=0.1)
        assert score.is_correct is False

    def test_is_passing_high_confidence(self):
        score = CorrectnessScore(verdict=CorrectnessVerdict.CORRECT, confidence=0.95)
        assert score.is_passing is True

    def test_is_not_passing_low_confidence(self):
        score = CorrectnessScore(verdict=CorrectnessVerdict.CORRECT, confidence=0.5)
        assert score.is_passing is False

    def test_is_not_passing_incorrect(self):
        score = CorrectnessScore(verdict=CorrectnessVerdict.INCORRECT, confidence=0.95)
        assert score.is_passing is False


class TestJudgeConfig:
    def test_defaults(self):
        config = JudgeConfig()
        assert config.use_cot is False
        assert config.deduplicate is False
        assert config.shuffle_claims is False
        assert config.shuffle_seed is None
        assert config.numerical_tolerance == 0.05

    def test_custom(self):
        config = JudgeConfig(
            use_cot=True,
            deduplicate=True,
            shuffle_claims=True,
            shuffle_seed=42,
            numerical_tolerance=0.03,
        )
        assert config.use_cot is True
        assert config.deduplicate is True
        assert config.shuffle_claims is True
        assert config.shuffle_seed == 42
        assert config.numerical_tolerance == 0.03


class TestDecomposedCorrectnessScore:
    def test_verdict_correct(self):
        score = DecomposedCorrectnessScore(
            precision=1.0,
            recall=0.95,
            f1=0.974,
            supported=[ClaimVerdict(claim="a", verdict="supported")],
        )
        assert score.verdict == CorrectnessVerdict.CORRECT

    def test_verdict_incorrect_low_f1(self):
        score = DecomposedCorrectnessScore(precision=0.2, recall=0.1, f1=0.13)
        assert score.verdict == CorrectnessVerdict.INCORRECT

    def test_verdict_incorrect_contradictions(self):
        score = DecomposedCorrectnessScore(
            precision=0.5,
            recall=0.5,
            f1=0.5,
            contradicted=[
                ClaimVerdict(claim="a", verdict="contradicted"),
                ClaimVerdict(claim="b", verdict="contradicted"),
            ],
            supported=[ClaimVerdict(claim="c", verdict="supported")],
        )
        assert score.verdict == CorrectnessVerdict.INCORRECT

    def test_verdict_partial(self):
        score = DecomposedCorrectnessScore(
            precision=0.7,
            recall=0.6,
            f1=0.65,
            supported=[ClaimVerdict(claim="a", verdict="supported")],
            contradicted=[ClaimVerdict(claim="b", verdict="contradicted")],
        )
        assert score.verdict == CorrectnessVerdict.PARTIALLY_CORRECT

    def test_to_correctness_score(self):
        score = DecomposedCorrectnessScore(
            precision=0.8,
            recall=0.9,
            f1=0.85,
            supported=[ClaimVerdict(claim="a", verdict="supported")],
            contradicted=[ClaimVerdict(claim="b", verdict="contradicted", evidence="wrong")],
            not_mentioned=[ClaimVerdict(claim="c", verdict="not_mentioned")],
        )
        cs = score.to_correctness_score()
        assert cs.confidence == 0.85
        assert len(cs.differences) == 2

    def test_is_passing(self):
        assert DecomposedCorrectnessScore(precision=1.0, recall=1.0, f1=0.85).is_passing is True
        assert DecomposedCorrectnessScore(precision=0.5, recall=0.5, f1=0.5).is_passing is False


class TestVitalScore:
    def test_verdict_correct(self):
        score = VitalScore(
            vital_precision=1.0,
            vital_recall=1.0,
            vital_f1=1.0,
            weighted_precision=0.9,
            weighted_recall=0.9,
            weighted_f1=0.9,
            response_level_precision=True,
            response_level_recall=True,
        )
        assert score.verdict == CorrectnessVerdict.CORRECT

    def test_verdict_fails_rlp(self):
        score = VitalScore(
            vital_precision=1.0,
            vital_recall=1.0,
            vital_f1=1.0,
            weighted_precision=1.0,
            weighted_recall=1.0,
            weighted_f1=1.0,
            response_level_precision=False,
            response_level_recall=True,
        )
        assert score.verdict == CorrectnessVerdict.INCORRECT

    def test_is_passing(self):
        passing = VitalScore(
            vital_precision=1.0,
            vital_recall=1.0,
            vital_f1=1.0,
            weighted_precision=1.0,
            weighted_recall=1.0,
            weighted_f1=1.0,
            response_level_precision=True,
            response_level_recall=True,
        )
        assert passing.is_passing is True

        failing = VitalScore(
            vital_precision=1.0,
            vital_recall=0.0,
            vital_f1=0.0,
            weighted_precision=1.0,
            weighted_recall=0.0,
            weighted_f1=0.0,
            response_level_precision=True,
            response_level_recall=False,
        )
        assert failing.is_passing is False


class TestNugget:
    def test_creation(self):
        n = Nugget(claim="test", importance=ClaimImportance.VITAL, weight=1.0)
        assert n.claim == "test"
        assert n.importance == ClaimImportance.VITAL

    def test_default_weight(self):
        n = Nugget(claim="test", importance=ClaimImportance.OKAY)
        assert n.weight == 1.0


class TestImportanceWeights:
    def test_vital(self):
        assert IMPORTANCE_WEIGHTS[ClaimImportance.VITAL] == 1.0

    def test_okay(self):
        assert IMPORTANCE_WEIGHTS[ClaimImportance.OKAY] == 0.5

    def test_not_important(self):
        assert IMPORTANCE_WEIGHTS[ClaimImportance.NOT_IMPORTANT] == 0.0
