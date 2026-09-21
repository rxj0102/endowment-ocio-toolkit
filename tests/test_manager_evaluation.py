import numpy as np
import pandas as pd
import pytest

from endowment_ocio_toolkit.manager_evaluation import (
    CashFlow,
    constant_growth_index,
    evaluate_liquid_manager,
    fund_multiples,
    generate_synthetic_manager_returns,
    generate_synthetic_pe_cash_flows,
    information_ratio,
    irr,
    kaplan_schoar_pme,
    rolling_sharpe_ratio,
    tracking_error,
    up_down_capture,
)


def test_irr_hand_checked_examples():
    assert irr([CashFlow(0, -100.0), CashFlow(1, 121.0)]) == pytest.approx(0.21, abs=1e-6)
    assert irr([CashFlow(0, -100.0), CashFlow(1, 100.0)]) == pytest.approx(0.0, abs=1e-6)


def test_irr_requires_both_signs():
    with pytest.raises(ValueError):
        irr([CashFlow(0, -100.0), CashFlow(1, -50.0)])


def test_fund_multiples_hand_checked():
    result = fund_multiples([CashFlow(0, -100.0), CashFlow(1, 50.0)], current_nav=70.0)
    assert result.dpi == pytest.approx(0.5)
    assert result.rvpi == pytest.approx(0.7)
    assert result.tvpi == pytest.approx(1.2)

    multi_flow = fund_multiples(
        [CashFlow(0, -40.0), CashFlow(1, -60.0), CashFlow(3, 30.0), CashFlow(4, 40.0)], current_nav=90.0
    )
    assert multi_flow.paid_in == pytest.approx(100.0)
    assert multi_flow.distributed == pytest.approx(70.0)
    assert multi_flow.tvpi == pytest.approx((70.0 + 90.0) / 100.0)


def test_fund_multiples_requires_positive_paid_in():
    with pytest.raises(ValueError):
        fund_multiples([CashFlow(0, 50.0)], current_nav=10.0)


def test_kaplan_schoar_pme_hand_checked():
    index = constant_growth_index(0.10)
    pme = kaplan_schoar_pme([CashFlow(0, -100.0), CashFlow(1, 150.0)], current_nav=0.0, index_level_at=index, valuation_period=1.0)
    assert pme == pytest.approx(150.0 / 110.0)

    outperform_index = constant_growth_index(0.05)
    strong_pme = kaplan_schoar_pme(
        [CashFlow(0, -100.0), CashFlow(2, 200.0)], current_nav=0.0, index_level_at=outperform_index, valuation_period=2.0
    )
    assert strong_pme > 1.0


def test_kaplan_schoar_pme_requires_positive_contributions():
    index = constant_growth_index(0.05)
    with pytest.raises(ValueError):
        kaplan_schoar_pme([CashFlow(0, 100.0)], current_nav=0.0, index_level_at=index, valuation_period=1.0)


def test_tracking_error_hand_checked():
    idx = pd.RangeIndex(24)
    flat = pd.Series(np.linspace(-0.02, 0.02, 24), index=idx)
    assert tracking_error(flat, flat) == pytest.approx(0.0, abs=1e-12)

    manager = pd.Series([0.01, 0.02, -0.01, 0.03])
    benchmark = pd.Series([0.00, 0.01, 0.00, 0.02])
    expected = (manager - benchmark).std(ddof=1) * np.sqrt(12)
    assert tracking_error(manager, benchmark, periods_per_year=12) == pytest.approx(expected)


def test_information_ratio_positive_when_manager_outperforms():
    rng = np.random.default_rng(0)
    idx = pd.RangeIndex(36)
    benchmark = pd.Series(rng.normal(0.005, 0.02, 36), index=idx)
    noise = pd.Series(rng.normal(0.0, 0.001, 36), index=idx)
    manager = benchmark + 0.002 + noise
    assert information_ratio(manager, benchmark, periods_per_year=12) > 0


def test_up_down_capture_hand_checked():
    benchmark = pd.Series([0.05, -0.05, 0.10, -0.10])
    manager = pd.Series([0.06, -0.04, 0.12, -0.08])
    up_capture, down_capture = up_down_capture(manager, benchmark)
    assert up_capture == pytest.approx(100.0 * ((0.06 + 0.12) / 2) / ((0.05 + 0.10) / 2))
    assert down_capture == pytest.approx(100.0 * ((-0.04 + -0.08) / 2) / ((-0.05 + -0.10) / 2))


def test_rolling_sharpe_ratio_behavior():
    constant_returns = pd.Series([0.01] * 10)
    rolling_constant = rolling_sharpe_ratio(constant_returns, window=5)
    assert np.isinf(rolling_constant.iloc[4:]).all(), "zero-volatility window should diverge"

    varying_returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005, 0.03, -0.02, 0.01, 0.02, 0.0])
    rolling_varying = rolling_sharpe_ratio(varying_returns, window=5)
    assert rolling_varying.iloc[4:].notna().all()


def test_evaluate_liquid_manager_report_fields():
    manager, benchmark = generate_synthetic_manager_returns(n_periods=48, seed=3)
    report = evaluate_liquid_manager(manager, benchmark)
    assert report.tracking_error > 0
    assert np.isfinite(report.information_ratio)
    assert np.isfinite(report.up_capture)
    assert np.isfinite(report.down_capture)


def test_synthetic_pe_cash_flows_are_valid_j_curves():
    cash_flows, nav = generate_synthetic_pe_cash_flows(seed=1, fund_life_years=10.0)
    calls = [cf for cf in cash_flows if cf.amount < 0]
    distributions = [cf for cf in cash_flows if cf.amount > 0]
    assert calls and distributions
    assert min(c.period for c in calls) < max(d.period for d in distributions)

    all_flows = cash_flows + ([CashFlow(10.0, nav)] if nav > 0 else [])
    assert np.isfinite(irr(all_flows))
    assert fund_multiples(cash_flows, current_nav=nav).tvpi > 0
