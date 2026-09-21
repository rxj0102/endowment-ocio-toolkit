import numpy as np
import pytest

from endowment_ocio_toolkit.spending_policy_model import (
    SpendingPolicyParams,
    quarterly_spend,
    simulate_endowment,
    summarize_simulation,
    trailing_average_market_value,
)


def test_trailing_average_full_window():
    values = np.array([100.0, 110.0, 90.0, 120.0])
    avg = trailing_average_market_value(values, smoothing_quarters=2)
    assert avg == pytest.approx([100.0, 105.0, 100.0, 105.0])


def test_trailing_average_uses_available_history_before_full_window():
    values = np.array([100.0, 200.0, 300.0])
    avg = trailing_average_market_value(values, smoothing_quarters=12)
    assert avg == pytest.approx([100.0, 150.0, 200.0])


def test_quarterly_spend_formula():
    spend = quarterly_spend(trailing_average=1000.0, annual_spend_rate=0.04)
    assert spend == pytest.approx(1000.0 * 0.04 / 4.0)


def test_simulate_endowment_shapes_and_reproducibility():
    result = simulate_endowment(0.06, 0.10, n_paths=50, horizon_years=5, seed=1)
    assert result.market_value_paths.shape == (50, 21)
    assert result.spending_paths.shape == (50, 20)
    assert result.real_value_paths.shape == result.market_value_paths.shape

    same_seed = simulate_endowment(0.06, 0.10, n_paths=50, horizon_years=5, seed=1)
    assert np.array_equal(result.market_value_paths, same_seed.market_value_paths)

    other_seed = simulate_endowment(0.06, 0.10, n_paths=50, horizon_years=5, seed=2)
    assert not np.array_equal(result.market_value_paths, other_seed.market_value_paths)


def test_probability_real_value_preserved_between_zero_and_one():
    result = simulate_endowment(0.06, 0.12, n_paths=200, horizon_years=10, seed=5)
    prob = result.probability_real_value_preserved(initial_value=100.0)
    assert 0.0 <= prob <= 1.0


def test_higher_return_improves_median_ending_value():
    low = simulate_endowment(0.04, 0.10, n_paths=500, horizon_years=15, seed=3)
    high = simulate_endowment(0.09, 0.10, n_paths=500, horizon_years=15, seed=3)
    assert high.median_ending_real_value() > low.median_ending_real_value()


def test_higher_volatility_increases_spending_volatility():
    low_vol = simulate_endowment(0.06, 0.05, n_paths=500, horizon_years=15, seed=4)
    high_vol = simulate_endowment(0.06, 0.20, n_paths=500, horizon_years=15, seed=4)
    assert high_vol.spending_volatility() > low_vol.spending_volatility()


def test_spending_never_exceeds_available_balance():
    result = simulate_endowment(0.02, 0.30, n_paths=300, horizon_years=10, seed=6)
    assert np.all(result.spending_paths <= result.market_value_paths[:, :-1] + 1e-9)


def test_summarize_simulation_keys():
    result = simulate_endowment(0.06, 0.10, n_paths=50, horizon_years=5, seed=1)
    summary = summarize_simulation(result, initial_value=100.0)
    expected_keys = {
        "probability_real_value_preserved",
        "median_ending_real_value",
        "spending_volatility",
        "median_ending_real_value_ratio",
    }
    assert set(summary.keys()) == expected_keys


def test_zero_inflation_makes_real_equal_nominal():
    params = SpendingPolicyParams(inflation_mean=0.0, inflation_vol=0.0)
    result = simulate_endowment(0.05, 0.08, params=params, n_paths=20, horizon_years=3, seed=8)
    assert np.allclose(result.market_value_paths, result.real_value_paths)
