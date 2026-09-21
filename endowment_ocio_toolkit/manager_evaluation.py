"""Manager due-diligence and performance evaluation for liquid and illiquid strategies.

DISCLAIMER: All manager return streams and fund cash flows in this module are
synthetically generated for demonstration only and do not represent any real
manager, fund, or track record.

Liquid strategies (e.g. long-only or hedge fund managers with monthly NAVs)
are evaluated with standard risk-adjusted metrics versus a benchmark: rolling
Sharpe ratio, tracking error, information ratio, and up/down capture.

Illiquid strategies (private equity/venture funds) are evaluated on their
J-curve cash flows: IRR, TVPI, DPI, RVPI, and a public market equivalent
(PME) benchmark using the Kaplan-Schoar (KS-PME) method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq

DISCLAIMER = (
    "SYNTHETIC DATA: manager return streams and fund cash flows in this "
    "module are fabricated for demonstration only."
)

_TRADING_PERIODS_PER_YEAR = 12  # monthly return streams throughout this module


# --------------------------------------------------------------------------
# Liquid strategy evaluation
# --------------------------------------------------------------------------


def rolling_sharpe_ratio(
    returns: pd.Series,
    window: int = 36,
    risk_free_rate: float = 0.0,
    periods_per_year: int = _TRADING_PERIODS_PER_YEAR,
) -> pd.Series:
    """Annualized rolling Sharpe ratio of a periodic return series.

    Args:
        returns: Periodic (e.g. monthly) manager returns.
        window: Rolling window length in periods.
        risk_free_rate: Annualized risk-free rate subtracted before annualizing.
        periods_per_year: Compounding periods per year (12 for monthly).
    """
    excess = returns - risk_free_rate / periods_per_year
    mean = excess.rolling(window).mean()
    std = excess.rolling(window).std(ddof=1)
    return (mean / std) * np.sqrt(periods_per_year)


def tracking_error(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = _TRADING_PERIODS_PER_YEAR,
) -> float:
    """Annualized standard deviation of active (manager - benchmark) returns."""
    active = returns - benchmark_returns
    return float(active.std(ddof=1) * np.sqrt(periods_per_year))


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = _TRADING_PERIODS_PER_YEAR,
) -> float:
    """Annualized mean active return divided by annualized tracking error."""
    active = returns - benchmark_returns
    te = tracking_error(returns, benchmark_returns, periods_per_year)
    if te == 0:
        return float("nan")
    annualized_active_return = float(active.mean() * periods_per_year)
    return annualized_active_return / te


def up_down_capture(returns: pd.Series, benchmark_returns: pd.Series) -> tuple[float, float]:
    """Up-market and down-market capture ratios (in percent) versus a benchmark.

    Up capture = average manager return in periods where the benchmark rose,
    divided by the average benchmark return in those same periods (x100).
    Down capture is the analogous ratio over periods where the benchmark fell.
    """
    up_mask = benchmark_returns > 0
    down_mask = benchmark_returns < 0

    up_capture = float("nan")
    if up_mask.any() and benchmark_returns[up_mask].mean() != 0:
        up_capture = 100.0 * returns[up_mask].mean() / benchmark_returns[up_mask].mean()

    down_capture = float("nan")
    if down_mask.any() and benchmark_returns[down_mask].mean() != 0:
        down_capture = 100.0 * returns[down_mask].mean() / benchmark_returns[down_mask].mean()

    return up_capture, down_capture


@dataclass
class LiquidManagerReport:
    rolling_sharpe: pd.Series
    tracking_error: float
    information_ratio: float
    up_capture: float
    down_capture: float


def evaluate_liquid_manager(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    rolling_window: int = 36,
    risk_free_rate: float = 0.0,
    periods_per_year: int = _TRADING_PERIODS_PER_YEAR,
) -> LiquidManagerReport:
    """Run the standard liquid-strategy due-diligence metric suite."""
    return LiquidManagerReport(
        rolling_sharpe=rolling_sharpe_ratio(returns, rolling_window, risk_free_rate, periods_per_year),
        tracking_error=tracking_error(returns, benchmark_returns, periods_per_year),
        information_ratio=information_ratio(returns, benchmark_returns, periods_per_year),
        up_capture=up_down_capture(returns, benchmark_returns)[0],
        down_capture=up_down_capture(returns, benchmark_returns)[1],
    )


def generate_synthetic_manager_returns(
    n_periods: int = 60,
    benchmark_annual_return: float = 0.08,
    benchmark_annual_vol: float = 0.15,
    manager_beta: float = 0.9,
    manager_annual_alpha: float = 0.01,
    manager_tracking_error: float = 0.04,
    periods_per_year: int = _TRADING_PERIODS_PER_YEAR,
    seed: int = 11,
) -> tuple[pd.Series, pd.Series]:
    """Generate a synthetic (manager, benchmark) monthly return pair for demos/tests."""
    rng = np.random.default_rng(seed)
    bench_mu = benchmark_annual_return / periods_per_year
    bench_sigma = benchmark_annual_vol / np.sqrt(periods_per_year)
    benchmark = rng.normal(bench_mu, bench_sigma, n_periods)

    idiosyncratic_sigma = manager_tracking_error / np.sqrt(periods_per_year)
    idiosyncratic = rng.normal(0.0, idiosyncratic_sigma, n_periods)
    manager = manager_beta * benchmark + manager_annual_alpha / periods_per_year + idiosyncratic

    index = pd.RangeIndex(n_periods)
    return pd.Series(manager, index=index, name="manager"), pd.Series(benchmark, index=index, name="benchmark")


# --------------------------------------------------------------------------
# Illiquid strategy (private equity / venture) evaluation
# --------------------------------------------------------------------------


@dataclass
class CashFlow:
    """One private-fund cash flow. Contributions are negative, distributions positive."""

    period: float  # in years from fund inception
    amount: float


def _npv(rate: float, cash_flows: list[CashFlow]) -> float:
    return sum(cf.amount / (1.0 + rate) ** cf.period for cf in cash_flows)


def irr(cash_flows: list[CashFlow], bracket: tuple[float, float] = (-0.9999, 10.0)) -> float:
    """Internal rate of return solving NPV(cash_flows, irr) = 0 via bisection.

    Requires at least one negative and one positive cash flow amount.
    """
    amounts = [cf.amount for cf in cash_flows]
    if not (any(a < 0 for a in amounts) and any(a > 0 for a in amounts)):
        raise ValueError("IRR requires both contributions (negative) and distributions/NAV (positive).")

    low, high = bracket
    f_low, f_high = _npv(low, cash_flows), _npv(high, cash_flows)
    if f_low * f_high > 0:
        raise ValueError("Could not bracket an IRR root in the given range; check cash flows.")
    return float(brentq(lambda r: _npv(r, cash_flows), low, high, xtol=1e-10))


@dataclass
class FundMultiples:
    tvpi: float
    dpi: float
    rvpi: float
    paid_in: float
    distributed: float
    residual_value: float


def fund_multiples(cash_flows: list[CashFlow], current_nav: float) -> FundMultiples:
    """Compute TVPI, DPI, and RVPI from a fund's contributions/distributions and current NAV.

    * DPI (Distributions to Paid-In) = cumulative distributions / cumulative paid-in capital
    * RVPI (Residual Value to Paid-In) = current NAV / cumulative paid-in capital
    * TVPI (Total Value to Paid-In) = DPI + RVPI
    """
    paid_in = -sum(cf.amount for cf in cash_flows if cf.amount < 0)
    distributed = sum(cf.amount for cf in cash_flows if cf.amount > 0)
    if paid_in <= 0:
        raise ValueError("Paid-in capital must be positive to compute multiples.")
    dpi = distributed / paid_in
    rvpi = current_nav / paid_in
    return FundMultiples(
        tvpi=dpi + rvpi,
        dpi=dpi,
        rvpi=rvpi,
        paid_in=paid_in,
        distributed=distributed,
        residual_value=current_nav,
    )


def kaplan_schoar_pme(
    cash_flows: list[CashFlow],
    current_nav: float,
    index_level_at: Callable[[float], float],
    valuation_period: float,
) -> float:
    """Kaplan-Schoar public market equivalent (KS-PME).

    Each fund cash flow is grown (or shrunk) by the public index's total
    return from the cash flow date to the valuation date, using
    ``index_level_at`` (a function mapping a period, in years, to an index
    level). A KS-PME above 1.0 means the fund outperformed investing the same
    dollars in the public index on the same schedule; below 1.0 means it
    underperformed.

    Args:
        cash_flows: Fund contributions (negative) and distributions (positive).
        current_nav: Fund's residual value as of ``valuation_period``.
        index_level_at: Callable giving the public market index level at a
            given period (in years from fund inception).
        valuation_period: The valuation date, in years from fund inception.
    """
    index_terminal = index_level_at(valuation_period)
    fv_contributions = 0.0
    fv_distributions = 0.0
    for cf in cash_flows:
        growth = index_terminal / index_level_at(cf.period)
        if cf.amount < 0:
            fv_contributions += -cf.amount * growth
        else:
            fv_distributions += cf.amount * growth
    if fv_contributions <= 0:
        raise ValueError("Contributions must be positive to compute a PME.")
    return (fv_distributions + current_nav) / fv_contributions


def generate_synthetic_pe_cash_flows(
    fund_size: float = 100.0,
    fund_life_years: float = 10.0,
    n_capital_calls: int = 6,
    n_distributions: int = 8,
    moic_target: float = 2.2,
    seed: int = 23,
) -> tuple[list[CashFlow], float]:
    """Generate a synthetic private-fund J-curve: front-loaded calls, back-loaded distributions.

    Returns the cash flow list and a plausible ending NAV (assumed fully
    realized: ending NAV is 0, i.e. the fund has fully distributed by
    ``fund_life_years``). Use a shorter ``fund_life_years`` slice of the
    output externally to model an interim (unrealized) valuation date.
    """
    rng = np.random.default_rng(seed)

    call_periods = np.sort(rng.uniform(0.0, fund_life_years * 0.4, n_capital_calls))
    call_weights = rng.dirichlet(np.ones(n_capital_calls) * 3.0)
    calls = [CashFlow(period=float(t), amount=-float(w * fund_size)) for t, w in zip(call_periods, call_weights)]

    dist_periods = np.sort(rng.uniform(fund_life_years * 0.35, fund_life_years, n_distributions))
    total_distributions = fund_size * moic_target
    dist_weights = rng.dirichlet(np.ones(n_distributions) * 2.0)
    distributions = [
        CashFlow(period=float(t), amount=float(w * total_distributions)) for t, w in zip(dist_periods, dist_weights)
    ]

    cash_flows = sorted(calls + distributions, key=lambda cf: cf.period)
    return cash_flows, 0.0


def constant_growth_index(annual_return: float, start_level: float = 100.0) -> Callable[[float], float]:
    """Build a deterministic ``index_level_at(period_years)`` growing at a constant annual rate."""

    def _level(period_years: float) -> float:
        return start_level * (1.0 + annual_return) ** period_years

    return _level
