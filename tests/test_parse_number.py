import pytest

from correctness_judge.base import CorrectnessJudge


parse = CorrectnessJudge._try_parse_number


class TestPlainNumbers:
    def test_integer(self):
        assert parse("42") == 42.0

    def test_negative(self):
        assert parse("-7.5") == -7.5

    def test_zero(self):
        assert parse("0") == 0.0

    def test_decimal(self):
        assert parse("3.14159") == pytest.approx(3.14159)


class TestCurrency:
    def test_dollar(self):
        assert parse("$45.99") == 45.99

    def test_euro(self):
        assert parse("\u20ac100") == 100.0

    def test_pound(self):
        assert parse("\u00a350") == 50.0

    def test_yen(self):
        assert parse("\u00a51000") == 1000.0


class TestCommas:
    def test_thousands(self):
        assert parse("1,234") == 1234.0

    def test_millions(self):
        assert parse("1,234,567") == 1234567.0

    def test_currency_with_commas(self):
        assert parse("$1,000,000") == 1000000.0


class TestPercentages:
    def test_integer_pct(self):
        assert parse("42%") == 42.0

    def test_decimal_pct(self):
        assert parse("3.5%") == 3.5


class TestSuffixes:
    def test_k(self):
        assert parse("5K") == 5000.0

    def test_m(self):
        assert parse("1.5m") == 1500000.0

    def test_b(self):
        assert parse("3.2B") == pytest.approx(3.2e9)

    def test_t(self):
        assert parse("1T") == 1e12

    def test_million_suffix(self):
        assert parse("1.5million") == 1500000.0


class TestSpacedWords:
    def test_million(self):
        assert parse("1.5 million") == 1500000.0

    def test_billion(self):
        assert parse("3 billion") == 3e9

    def test_trillion(self):
        assert parse("2.7 trillion") == pytest.approx(2.7e12)

    def test_thousand(self):
        assert parse("500 thousand") == 500000.0


class TestNonNumeric:
    def test_word(self):
        assert parse("hello") is None

    def test_sentence(self):
        assert parse("not a number") is None

    def test_empty(self):
        assert parse("") is None

    def test_whitespace(self):
        assert parse("   ") is None
