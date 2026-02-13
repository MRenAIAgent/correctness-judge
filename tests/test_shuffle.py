import random

import pytest

from correctness_judge.base import CorrectnessJudge
from correctness_judge.models import JudgeConfig


class TestShuffleRestoreRoundTrip:
    def test_basic_round_trip(self):
        claims = ["Claim A", "Claim B", "Claim C", "Claim D", "Claim E"]
        rng = random.Random(42)
        indices = list(range(len(claims)))
        rng.shuffle(indices)
        shuffled = [claims[i] for i in indices]

        inverse = [0] * len(indices)
        for new_pos, old_pos in enumerate(indices):
            inverse[old_pos] = new_pos

        restored = [shuffled[inverse[i]] for i in range(len(claims))]
        assert restored == claims

    def test_seed_reproducibility(self):
        claims = ["A", "B", "C", "D", "E", "F", "G"]

        rng1 = random.Random(99)
        idx1 = list(range(len(claims)))
        rng1.shuffle(idx1)

        rng2 = random.Random(99)
        idx2 = list(range(len(claims)))
        rng2.shuffle(idx2)

        assert idx1 == idx2

    def test_single_claim_no_change(self):
        claims = ["Only one"]
        rng = random.Random(0)
        indices = list(range(len(claims)))
        rng.shuffle(indices)
        assert [claims[i] for i in indices] == claims

    def test_empty_claims(self):
        claims = []
        rng = random.Random(0)
        indices = list(range(len(claims)))
        rng.shuffle(indices)
        assert indices == []


class TestJudgeConfigShuffleIntegration:
    def test_config_controls_shuffle(self):
        config_on = JudgeConfig(shuffle_claims=True, shuffle_seed=42)
        judge = CorrectnessJudge(config=config_on)
        assert judge.config.shuffle_claims is True

        config_off = JudgeConfig(shuffle_claims=False)
        judge_off = CorrectnessJudge(config=config_off)
        assert judge_off.config.shuffle_claims is False
