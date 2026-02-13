from correctness_judge.base import CorrectnessJudge


judge = CorrectnessJudge()


class TestParseJson:
    def test_plain_json(self):
        result = judge._parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_code_fence(self):
        result = judge._parse_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_json_with_plain_fence(self):
        result = judge._parse_json('```\n{"key": 42}\n```')
        assert result == {"key": 42}

    def test_json_with_surrounding_text(self):
        result = judge._parse_json('Here is the result: {"key": "value"} done')
        assert result == {"key": "value"}

    def test_empty_string(self):
        result = judge._parse_json("")
        assert result == {}

    def test_invalid_json(self):
        result = judge._parse_json("not json at all")
        assert result == {}

    def test_nested_json(self):
        text = '{"verdicts": [{"claim": "test", "verdict": "supported"}]}'
        result = judge._parse_json(text)
        assert len(result["verdicts"]) == 1
        assert result["verdicts"][0]["verdict"] == "supported"


class TestParseResponse:
    def test_correct_verdict(self):
        text = '{"verdict": "correct", "confidence": 0.95, "differences": [], "explanation": "all good"}'
        score = judge._parse_response(text)
        assert score.verdict.value == "correct"
        assert score.confidence == 0.95

    def test_sycophancy_guard_downgrades(self):
        text = '{"verdict": "correct", "confidence": 0.95, "differences": ["found a diff"], "explanation": "mostly good"}'
        score = judge._parse_response(text)
        assert score.verdict.value == "partially_correct"
        assert score.confidence <= 0.7

    def test_invalid_verdict_defaults_incorrect(self):
        text = '{"verdict": "maybe", "confidence": 0.5, "differences": [], "explanation": "unsure"}'
        score = judge._parse_response(text)
        assert score.verdict.value == "incorrect"

    def test_unparseable_response(self):
        score = judge._parse_response("completely unparseable garbage")
        assert score.verdict.value == "incorrect"
        assert score.confidence == 0.0
