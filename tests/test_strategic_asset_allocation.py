import numpy as np
import pandas as pd
import pytest

from endowment_ocio_toolkit.capital_market_assumptions import build_cma
from endowment_ocio_toolkit.strategic_asset_allocation import (
    HouseView,
    PortfolioConstraints,
    black_litterman_posterior,
    build_policy_portfolios,
    compare_policy_portfolios,
    efficient_frontier,
    expected_max_drawdown,
    implied_equilibrium_returns,
    max_return_portfolio,
    min_variance_portfolio,
    portfolio_return,
    portfolio_volatility,
)


@pytest.fixture(scope="module")
def cma():
    return build_cma()


@pytest.fixture(scope="module")
def constraints():
    return PortfolioConstraints(max_private_equity=0.15, max_illiquid_total=0.35, min_liquidity_buffer=0.03)


def test_min_variance_portfolio_is_feasible(cma, constraints):
    w = min_variance_portfolio(cma, constraints)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-8).all(), "no-short constraint violated"
    assert w["Cash"] >= constraints.min_liquidity_buffer - 1e-6


def test_max_return_portfolio_respects_illiquidity_caps(cma, constraints):
    w = max_return_portfolio(cma, constraints)
    assert w["Private Equity / Venture"] <= constraints.max_private_equity + 1e-6
    illiquid_total = w["Private Equity / Venture"] + w["Hedge Funds - Diversifying"] + w["Hedge Funds - Directional"]
    assert illiquid_total <= constraints.max_illiquid_total + 1e-6


def test_efficient_frontier_respects_constraints(cma, constraints):
    frontier = efficient_frontier(cma, constraints, n_points=8)
    weight_cols = cma.names
    assert np.allclose(frontier[weight_cols].sum(axis=1), 1.0, atol=1e-6), "weights must sum to 1"
    assert (frontier[weight_cols] >= -1e-6).all().all(), "no-short constraint violated"
    illiquid_cols = ["Private Equity / Venture", "Hedge Funds - Diversifying", "Hedge Funds - Directional"]
    assert (frontier[illiquid_cols].sum(axis=1) <= constraints.max_illiquid_total + 1e-6).all()


def test_efficient_frontier_volatility_is_non_decreasing_in_return(cma, constraints):
    frontier = efficient_frontier(cma, constraints, n_points=15).sort_values("expected_return")
    vols = frontier["volatility"].to_numpy()
    assert np.all(np.diff(vols) >= -1e-6)


def test_portfolio_return_and_volatility_helpers(cma):
    w = np.full(len(cma.names), 1.0 / len(cma.names))
    mu = cma.expected_returns.to_numpy()
    cov = cma.covariance.to_numpy()
    assert portfolio_return(w, mu) == pytest.approx(float(w @ mu))
    assert portfolio_volatility(w, cov) == pytest.approx(float(np.sqrt(w @ cov @ w)))


def test_policy_portfolios_ordered_by_risk_and_return(cma, constraints):
    portfolios = build_policy_portfolios(cma, constraints)
    assert portfolios["Conservative"].volatility < portfolios["Moderate"].volatility < portfolios["Growth"].volatility
    assert portfolios["Conservative"].expected_return <= portfolios["Moderate"].expected_return
    assert portfolios["Moderate"].expected_return <= portfolios["Growth"].expected_return


def test_policy_portfolio_drawdowns_negative_and_worsen_with_risk(cma, constraints):
    portfolios = build_policy_portfolios(cma, constraints)
    for p in portfolios.values():
        assert p.expected_max_drawdown < 0
    assert portfolios["Growth"].expected_max_drawdown < portfolios["Conservative"].expected_max_drawdown


def test_compare_policy_portfolios_shape(cma, constraints):
    portfolios = build_policy_portfolios(cma, constraints)
    table = compare_policy_portfolios(portfolios)
    assert set(table.index) == {"Conservative", "Moderate", "Growth"}
    assert set(table.columns) == {"expected_return", "volatility", "sharpe_ratio", "expected_max_drawdown"}


def test_expected_max_drawdown_properties():
    a = expected_max_drawdown(0.06, 0.12, seed=99)
    b = expected_max_drawdown(0.06, 0.12, seed=99)
    assert a == b, "should be reproducible given a fixed seed"
    low_vol = expected_max_drawdown(0.06, 0.05, seed=1)
    high_vol = expected_max_drawdown(0.06, 0.25, seed=1)
    assert high_vol < low_vol, "higher volatility should produce a deeper expected drawdown"


def test_black_litterman_no_views_returns_prior(cma):
    w = pd.Series(1.0 / len(cma.names), index=cma.names)
    prior = implied_equilibrium_returns(cma, w)
    result = black_litterman_posterior(cma, w, views=[])
    pd.testing.assert_series_equal(prior, result.prior_returns, check_names=False)
    pd.testing.assert_series_equal(result.prior_returns, result.posterior_returns, check_names=False)


def test_black_litterman_view_and_confidence_effects(cma):
    w = pd.Series(1.0 / len(cma.names), index=cma.names)
    prior = implied_equilibrium_returns(cma, w)
    target = prior["Emerging Markets Equity"] + 0.10

    bullish_view = HouseView(["Emerging Markets Equity"], [1.0], target, confidence=0.95)
    result = black_litterman_posterior(cma, w, views=[bullish_view])
    assert result.posterior_returns["Emerging Markets Equity"] > result.prior_returns["Emerging Markets Equity"]

    low_conf = black_litterman_posterior(cma, w, views=[HouseView(["Emerging Markets Equity"], [1.0], target, 0.2)])
    high_conf = black_litterman_posterior(cma, w, views=[HouseView(["Emerging Markets Equity"], [1.0], target, 0.95)])
    low_gap = abs(target - low_conf.posterior_returns["Emerging Markets Equity"])
    high_gap = abs(target - high_conf.posterior_returns["Emerging Markets Equity"])
    assert high_gap < low_gap, "higher-confidence views should pull the posterior closer to the view"
