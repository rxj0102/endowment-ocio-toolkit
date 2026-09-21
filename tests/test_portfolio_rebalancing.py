import numpy as np
import pandas as pd
import pytest

from endowment_ocio_toolkit.portfolio_rebalancing import (
    RebalancingPolicy,
    compare_rebalancing_policies,
    portfolio_returns_from_weights,
    simulate_drift,
    tracking_error_to_target,
)

NAMES = ["US Equity", "Core Fixed Income", "Private Equity / Venture"]


@pytest.fixture
def target_weights():
    return pd.Series([0.5, 0.3, 0.2], index=NAMES)


@pytest.fixture
def divergent_returns():
    # US Equity consistently strong, Core Fixed Income weak, PE flat: drift is guaranteed.
    n = 20
    return pd.DataFrame(
        {
            "US Equity": [0.04] * n,
            "Core Fixed Income": [-0.01] * n,
            "Private Equity / Venture": [0.0] * n,
        }
    )


def test_weights_always_sum_to_one(target_weights, divergent_returns):
    result = simulate_drift(target_weights, divergent_returns, illiquid_assets=["Private Equity / Venture"])
    for frame in result.values():
        sums = frame.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-9)


def test_no_rebalance_weights_drift_away_from_target(target_weights, divergent_returns):
    result = simulate_drift(target_weights, divergent_returns, illiquid_assets=["Private Equity / Venture"])
    final = result["no_rebalance"].iloc[-1]
    assert final["US Equity"] > target_weights["US Equity"]
    assert final["Core Fixed Income"] < target_weights["Core Fixed Income"]


def test_rebalanced_weights_stay_within_band_after_trigger(target_weights, divergent_returns):
    policy = RebalancingPolicy(band=0.02, rebalance_check_frequency=1)
    result = simulate_drift(
        target_weights, divergent_returns, illiquid_assets=["Private Equity / Venture"], policy=policy
    )
    rebalanced = result["rebalanced"]
    # After the first rebalance, liquid-sleeve deviation from target should reset near zero.
    liquid_cols = ["US Equity", "Core Fixed Income"]
    max_dev_after_first_period = (rebalanced.iloc[2][liquid_cols] - target_weights[liquid_cols]).abs().max()
    assert max_dev_after_first_period < 0.10


def test_illiquid_asset_dollar_value_only_compounds_by_its_own_return(target_weights, divergent_returns):
    """Rebalancing reallocates the liquid sleeve, so PE's *weight* path can differ
    between scenarios (total portfolio value grows at a different rate). What must
    hold regardless is that PE's *dollar* value is never itself traded: it only
    ever compounds by its own return, in both the rebalanced and no-rebalance runs.
    """
    policy = RebalancingPolicy(band=0.02, rebalance_check_frequency=1)
    result = simulate_drift(
        target_weights, divergent_returns, illiquid_assets=["Private Equity / Venture"], policy=policy
    )
    expected_growth = 1.0 + divergent_returns["Private Equity / Venture"].to_numpy()

    for scenario in ("rebalanced", "no_rebalance"):
        weights = result[scenario]
        port_returns = portfolio_returns_from_weights(weights, divergent_returns)
        wealth_multiplier = np.concatenate([[1.0], np.cumprod(1.0 + port_returns.to_numpy())])
        pe_dollar = weights["Private Equity / Venture"].to_numpy() * wealth_multiplier
        implied_growth = pe_dollar[1:] / pe_dollar[:-1]
        np.testing.assert_allclose(implied_growth, expected_growth, atol=1e-9)


def test_tracking_error_is_zero_when_returns_are_uniform_across_assets(target_weights):
    n = 10
    uniform_returns = pd.DataFrame({name: [0.01] * n for name in NAMES})
    result = simulate_drift(target_weights, uniform_returns, illiquid_assets=["Private Equity / Venture"])
    te = tracking_error_to_target(result["no_rebalance"], target_weights, uniform_returns)
    assert te == pytest.approx(0.0, abs=1e-10)


def test_rebalancing_reduces_or_matches_tracking_error(target_weights, divergent_returns):
    policy = RebalancingPolicy(band=0.02, rebalance_check_frequency=1)
    comparison = compare_rebalancing_policies(
        target_weights, divergent_returns, illiquid_assets=["Private Equity / Venture"], policy=policy
    )
    assert comparison.rebalanced_tracking_error <= comparison.no_rebalance_tracking_error + 1e-9


def test_cash_flow_based_rebalancing_directs_new_cash_to_underweight_assets(target_weights, divergent_returns):
    n = len(divergent_returns)
    # Positive external contribution every period, on top of return-driven drift.
    flows = pd.Series([0.05] * n)
    policy = RebalancingPolicy(band=1.0, rebalance_check_frequency=n + 1)  # disable band trigger
    result = simulate_drift(
        target_weights,
        divergent_returns,
        illiquid_assets=["Private Equity / Venture"],
        policy=policy,
        external_cash_flows=flows,
    )
    with_flows_final = result["rebalanced"].iloc[-1]

    no_flows = simulate_drift(
        target_weights, divergent_returns, illiquid_assets=["Private Equity / Venture"], policy=policy
    )["rebalanced"].iloc[-1]

    # Cash-flow-based rebalancing should leave Core Fixed Income (the laggard,
    # underweight asset) closer to target than doing nothing at all.
    fi_gap_with_flows = abs(with_flows_final["Core Fixed Income"] - target_weights["Core Fixed Income"])
    fi_gap_without_flows = abs(no_flows["Core Fixed Income"] - target_weights["Core Fixed Income"])
    assert fi_gap_with_flows < fi_gap_without_flows
