"""Unit tests for comprag.aggregate: bootstrap_ci and compute_preference_gap.

Tests-first: these will fail with ImportError until comprag/aggregate.py is built.
"""

import numpy as np
import pytest

from comprag.aggregate import bootstrap_ci, compute_preference_gap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(query_id: str, cu: float) -> dict:
    """Minimal scored record with the fields compute_preference_gap needs."""
    return {
        "query_id": query_id,
        "scores": {
            "ragchecker": {
                "context_utilization": cu,
                "self_knowledge": 0.1,
                "noise_sensitivity": 0.05,
            }
        },
    }


# ===========================================================================
# bootstrap_ci
# ===========================================================================

class TestBootstrapCI:
    """Tests for bootstrap_ci(values, n_resamples, confidence)."""

    def test_known_distribution_mean_in_range(self) -> None:
        """Mean of a known uniform distribution falls within expected range."""
        rng = np.random.default_rng(42)
        values = rng.uniform(0.0, 1.0, size=500)
        mean, ci_lo, ci_hi = bootstrap_ci(values, n_resamples=2000)
        # True mean of Uniform(0,1) is 0.5; sample mean should be close.
        assert 0.4 < mean < 0.6, f"Mean {mean} outside expected range"

    def test_ci_contains_true_mean(self) -> None:
        """95% CI from a normal distribution should contain the true mean."""
        rng = np.random.default_rng(99)
        true_mean = 5.0
        values = rng.normal(loc=true_mean, scale=1.0, size=200)
        mean, ci_lo, ci_hi = bootstrap_ci(values, n_resamples=5000, confidence=0.95)
        assert ci_lo <= true_mean <= ci_hi, (
            f"True mean {true_mean} not in CI [{ci_lo}, {ci_hi}]"
        )

    def test_ci_ordering(self) -> None:
        """ci_lo <= mean <= ci_hi always holds."""
        rng = np.random.default_rng(7)
        values = rng.normal(loc=0.0, scale=2.0, size=100)
        mean, ci_lo, ci_hi = bootstrap_ci(values)
        assert ci_lo <= mean <= ci_hi

    def test_ci_width_reasonable(self) -> None:
        """CI width for a tight distribution should be small."""
        rng = np.random.default_rng(0)
        values = rng.normal(loc=10.0, scale=0.01, size=500)
        mean, ci_lo, ci_hi = bootstrap_ci(values, n_resamples=2000)
        width = ci_hi - ci_lo
        assert width < 0.1, f"CI width {width} unexpectedly large for tight distribution"

    def test_wider_confidence_gives_wider_ci(self) -> None:
        """99% CI should be at least as wide as 90% CI."""
        rng = np.random.default_rng(12)
        values = rng.normal(loc=0.0, scale=1.0, size=300)
        _, lo_90, hi_90 = bootstrap_ci(values, n_resamples=3000, confidence=0.90)
        _, lo_99, hi_99 = bootstrap_ci(values, n_resamples=3000, confidence=0.99)
        assert (hi_99 - lo_99) >= (hi_90 - lo_90) * 0.9  # allow small jitter

    def test_returns_tuple_of_three_floats(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bootstrap_ci(values)
        assert isinstance(result, tuple)
        assert len(result) == 3
        for v in result:
            assert isinstance(v, float)

    # --- Edge cases ---

    def test_all_same_values(self) -> None:
        """All identical values -> mean equals that value, CI width is zero."""
        values = np.full(50, 0.75)
        mean, ci_lo, ci_hi = bootstrap_ci(values)
        assert mean == pytest.approx(0.75)
        assert ci_lo == pytest.approx(0.75)
        assert ci_hi == pytest.approx(0.75)

    def test_single_value(self) -> None:
        """Single-element array -> mean equals the value, degenerate CI."""
        values = np.array([3.14])
        mean, ci_lo, ci_hi = bootstrap_ci(values)
        assert mean == pytest.approx(3.14)
        # With one value, every resample is the same -> CI collapses.
        assert ci_lo == pytest.approx(3.14)
        assert ci_hi == pytest.approx(3.14)

    def test_empty_array_raises(self) -> None:
        """Empty array should raise ValueError."""
        with pytest.raises((ValueError, IndexError)):
            bootstrap_ci(np.array([]))


# ===========================================================================
# compute_preference_gap
# ===========================================================================

class TestComputePreferenceGap:
    """Tests for compute_preference_gap(pass2_records, pass3_records)."""

    def test_positive_gap(self) -> None:
        """pass3 CU > pass2 CU -> positive preference gap."""
        pass2 = [_make_record("q1", 0.5), _make_record("q2", 0.4)]
        pass3 = [_make_record("q1", 0.8), _make_record("q2", 0.7)]
        result = compute_preference_gap(pass2, pass3)
        assert result["mean"] > 0, f"Expected positive gap, got {result['mean']}"

    def test_negative_gap(self) -> None:
        """pass3 CU < pass2 CU -> negative preference gap."""
        pass2 = [_make_record("q1", 0.9), _make_record("q2", 0.8)]
        pass3 = [_make_record("q1", 0.3), _make_record("q2", 0.2)]
        result = compute_preference_gap(pass2, pass3)
        assert result["mean"] < 0, f"Expected negative gap, got {result['mean']}"

    def test_zero_gap(self) -> None:
        """Identical CU in pass2 and pass3 -> zero preference gap."""
        pass2 = [_make_record("q1", 0.6), _make_record("q2", 0.7)]
        pass3 = [_make_record("q1", 0.6), _make_record("q2", 0.7)]
        result = compute_preference_gap(pass2, pass3)
        assert result["mean"] == pytest.approx(0.0, abs=1e-10)

    def test_correct_per_query_computation(self) -> None:
        """Verify the per-query difference is pass3_cu - pass2_cu."""
        pass2 = [
            _make_record("q1", 0.3),
            _make_record("q2", 0.5),
            _make_record("q3", 0.7),
        ]
        pass3 = [
            _make_record("q1", 0.6),  # diff = +0.3
            _make_record("q2", 0.4),  # diff = -0.1
            _make_record("q3", 0.9),  # diff = +0.2
        ]
        result = compute_preference_gap(pass2, pass3)
        # Expected mean of diffs: (0.3 + (-0.1) + 0.2) / 3 = 0.1333...
        expected_mean = (0.3 + (-0.1) + 0.2) / 3.0
        assert result["mean"] == pytest.approx(expected_mean, abs=0.05)

    def test_output_keys(self) -> None:
        """Output dict must have exactly mean, ci_lo, ci_hi, std."""
        pass2 = [_make_record("q1", 0.5)]
        pass3 = [_make_record("q1", 0.7)]
        result = compute_preference_gap(pass2, pass3)
        required_keys = {"mean", "ci_lo", "ci_hi", "std"}
        assert required_keys == set(result.keys()), (
            f"Expected keys {required_keys}, got {set(result.keys())}"
        )

    def test_output_values_are_floats(self) -> None:
        """All output values should be floats."""
        pass2 = [_make_record("q1", 0.5), _make_record("q2", 0.6)]
        pass3 = [_make_record("q1", 0.7), _make_record("q2", 0.8)]
        result = compute_preference_gap(pass2, pass3)
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is {type(val)}, expected float"

    def test_ci_ordering_in_output(self) -> None:
        """ci_lo <= mean <= ci_hi in the preference gap output."""
        rng = np.random.default_rng(55)
        n = 50
        pass2 = [_make_record(f"q{i}", float(rng.uniform(0.3, 0.6))) for i in range(n)]
        pass3 = [_make_record(f"q{i}", float(rng.uniform(0.5, 0.9))) for i in range(n)]
        result = compute_preference_gap(pass2, pass3)
        assert result["ci_lo"] <= result["mean"] <= result["ci_hi"]

    def test_std_non_negative(self) -> None:
        """Standard deviation must be non-negative."""
        pass2 = [_make_record("q1", 0.5), _make_record("q2", 0.6)]
        pass3 = [_make_record("q1", 0.8), _make_record("q2", 0.3)]
        result = compute_preference_gap(pass2, pass3)
        assert result["std"] >= 0.0

    def test_many_queries_bootstrap_converges(self) -> None:
        """With many queries, bootstrap CI should be tight around the true gap."""
        rng = np.random.default_rng(77)
        n = 200
        true_gap = 0.15
        pass2 = [_make_record(f"q{i}", 0.5) for i in range(n)]
        pass3 = [
            _make_record(f"q{i}", 0.5 + true_gap + float(rng.normal(0, 0.02)))
            for i in range(n)
        ]
        result = compute_preference_gap(pass2, pass3)
        assert result["mean"] == pytest.approx(true_gap, abs=0.03)
        ci_width = result["ci_hi"] - result["ci_lo"]
        assert ci_width < 0.1, f"CI width {ci_width} too large for n={n}"
