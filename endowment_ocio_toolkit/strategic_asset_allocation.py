"""Mean-variance / Black-Litterman strategic asset allocation (SAA) engine.

DISCLAIMER: This module operates on the synthetic capital market assumptions in
``capital_market_assumptions.py``. Outputs are for demonstration only and are
not investment advice.

The optimizer uses ``scipy.optimize.minimize`` (SLSQP) to build an efficient
frontier subject to realistic institutional constraints:

* no shorting (weights >= 0)
* an illiquidity cap on the combined private-equity/venture and hedge-fund
  sleeve (capital that cannot be redeemed on demand)
* a separate, tighter cap on private equity/venture alone
* a minimum liquidity buffer held in cash

A Black-Litterman layer lets "house views" tilt the market-implied
(reverse-optimized) equilibrium prior before the frontier is built.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .capital_market_assumptions import CapitalMarketAssumptions

DISCLAIMER = (
    "SYNTHETIC DATA: strategic asset allocation outputs are illustrative only "
    "and must not be used to make real investment decisions."
)


# --------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioConstraints:
    """Institutional constraints applied to every optimization in this module."""

    max_private_equity: float = 0.15
    max_illiquid_total: float = 0.35
    min_liquidity_buffer: float = 0.03
    allow_short: bool = False

    def pe_mask(self, cma: CapitalMarketAssumptions) -> np.ndarray:
        return np.array(["Private Equity" in n for n in cma.names])

    def illiquid_mask(self, cma: CapitalMarketAssumptions) -> np.ndarray:
        return np.array(["Private Equity" in n or "Hedge Fund" in n for n in cma.names])

    def cash_mask(self, cma: CapitalMarketAssumptions) -> np.ndarray:
        return np.array([n == "Cash" for n in cma.names])


def _bounds(n: int, allow_short: bool) -> list[tuple[float, float]]:
    return [(0.0, 1.0)] * n if not allow_short else [(-1.0, 1.0)] * n


def _inequality_constraints(cma: CapitalMarketAssumptions, constraints: PortfolioConstraints) -> list[dict]:
    pe_mask = constraints.pe_mask(cma).astype(float)
    illiquid_mask = constraints.illiquid_mask(cma).astype(float)
    cash_mask = constraints.cash_mask(cma).astype(float)
    return [
        {"type": "ineq", "fun": lambda w: constraints.max_private_equity - pe_mask @ w},
        {"type": "ineq", "fun": lambda w: constraints.max_illiquid_total - illiquid_mask @ w},
        {"type": "ineq", "fun": lambda w: cash_mask @ w - constraints.min_liquidity_buffer},
    ]


def _equality_constraint() -> dict:
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    return float(np.dot(weights, mu))


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    variance = float(weights @ cov @ weights)
    return float(np.sqrt(max(variance, 0.0)))


def _feasible_start(n: int, cma: CapitalMarketAssumptions, constraints: PortfolioConstraints) -> np.ndarray:
    w = np.full(n, 1.0 / n)
    cash_idx = np.where(constraints.cash_mask(cma))[0]
    if cash_idx.size and w[cash_idx[0]] < constraints.min_liquidity_buffer:
        deficit = constraints.min_liquidity_buffer - w[cash_idx[0]]
        w[cash_idx[0]] += deficit
        w = w / w.sum()
    return w


def min_variance_portfolio(cma: CapitalMarketAssumptions, constraints: PortfolioConstraints) -> pd.Series:
    """Global minimum-variance portfolio subject to the institutional constraints."""
    mu = cma.expected_returns.to_numpy()
    cov = cma.covariance.to_numpy()
    n = len(mu)
    result = minimize(
        lambda w: portfolio_volatility(w, cov) ** 2,
        x0=_feasible_start(n, cma, constraints),
        method="SLSQP",
        bounds=_bounds(n, constraints.allow_short),
        constraints=[_equality_constraint(), *_inequality_constraints(cma, constraints)],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Min-variance optimization failed: {result.message}")
    return pd.Series(result.x, index=cma.names, name="weight")


def max_return_portfolio(cma: CapitalMarketAssumptions, constraints: PortfolioConstraints) -> pd.Series:
    """Maximum-expected-return portfolio subject to the institutional constraints."""
    mu = cma.expected_returns.to_numpy()
    n = len(mu)
    result = minimize(
        lambda w: -portfolio_return(w, mu),
        x0=_feasible_start(n, cma, constraints),
        method="SLSQP",
        bounds=_bounds(n, constraints.allow_short),
        constraints=[_equality_constraint(), *_inequality_constraints(cma, constraints)],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Max-return optimization failed: {result.message}")
    return pd.Series(result.x, index=cma.names, name="weight")


def efficient_frontier(
    cma: CapitalMarketAssumptions,
    constraints: PortfolioConstraints | None = None,
    n_points: int = 25,
    expected_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Build an efficient frontier: minimum variance for each of ``n_points`` target returns.

    Args:
        cma: Capital market assumptions supplying volatilities/correlations.
        constraints: Institutional constraints (defaults applied if omitted).
        n_points: Number of frontier points to solve for.
        expected_returns: Optional override of ``cma.expected_returns`` (e.g. a
            Black-Litterman posterior) used as the return vector.

    Returns:
        DataFrame indexed by point number with columns ``expected_return``,
        ``volatility``, ``sharpe_ratio`` and one weight column per asset class.
    """
    constraints = constraints or PortfolioConstraints()
    mu = (expected_returns if expected_returns is not None else cma.expected_returns).to_numpy()
    cov = cma.covariance.to_numpy()
    n = len(mu)
    cash_return = float(cma.expected_returns.get("Cash", 0.0))

    min_var_w = min_variance_portfolio(cma, constraints).to_numpy()
    max_ret_w = max_return_portfolio(cma, constraints).to_numpy()
    r_low = portfolio_return(min_var_w, mu)
    r_high = portfolio_return(max_ret_w, mu)
    if r_high < r_low:
        r_low, r_high = r_high, r_low
    target_returns = np.linspace(r_low, r_high, n_points)

    rows = []
    x0 = min_var_w.copy()
    for target in target_returns:
        cons = [
            _equality_constraint(),
            {"type": "eq", "fun": lambda w, t=target: portfolio_return(w, mu) - t},
            *_inequality_constraints(cma, constraints),
        ]
        result = minimize(
            lambda w: portfolio_volatility(w, cov) ** 2,
            x0=x0,
            method="SLSQP",
            bounds=_bounds(n, constraints.allow_short),
            constraints=cons,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not result.success:
            continue
        x0 = result.x
        vol = portfolio_volatility(result.x, cov)
        row = {
            "expected_return": target,
            "volatility": vol,
            "sharpe_ratio": (target - cash_return) / vol if vol > 0 else np.nan,
        }
        row.update({name: w for name, w in zip(cma.names, result.x)})
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Black-Litterman
# --------------------------------------------------------------------------


@dataclass
class HouseView:
    """One Black-Litterman view.

    For an absolute view, ``assets`` has one name and ``view_weights`` is
    ``[1.0]``. For a relative view (asset A will outperform asset B by
    ``magnitude``), use ``assets=[A, B]`` and ``view_weights=[1.0, -1.0]``.
    """

    assets: list[str]
    view_weights: list[float]
    magnitude: float
    confidence: float = 0.5  # in (0, 1]; 1.0 = dogmatic view


@dataclass
class BlackLittermanResult:
    prior_returns: pd.Series
    posterior_returns: pd.Series
    views: list[HouseView] = field(default_factory=list)


def implied_equilibrium_returns(
    cma: CapitalMarketAssumptions,
    market_weights: pd.Series,
    risk_aversion: float = 2.5,
) -> pd.Series:
    """Reverse-optimize the market-implied equilibrium return vector (the BL prior)."""
    w = market_weights.reindex(cma.names).to_numpy()
    cov = cma.covariance.to_numpy()
    pi = risk_aversion * cov @ w
    return pd.Series(pi, index=cma.names, name="implied_return")


def _views_to_matrices(cma: CapitalMarketAssumptions, views: list[HouseView]) -> tuple[np.ndarray, np.ndarray]:
    n = len(cma.names)
    idx = {name: i for i, name in enumerate(cma.names)}
    P = np.zeros((len(views), n))
    Q = np.zeros(len(views))
    for k, view in enumerate(views):
        for asset, weight in zip(view.assets, view.view_weights):
            P[k, idx[asset]] = weight
        Q[k] = view.magnitude
    return P, Q


def black_litterman_posterior(
    cma: CapitalMarketAssumptions,
    market_weights: pd.Series,
    views: list[HouseView],
    risk_aversion: float = 2.5,
    tau: float = 0.05,
) -> BlackLittermanResult:
    """Blend house views into the market-implied prior via the Black-Litterman formula.

    ``Omega`` (view uncertainty) is derived from each view's stated confidence
    following the He-Litterman convention: ``Omega_kk = (1/confidence - 1) *
    (P @ (tau * Sigma) @ P.T)_kk``, so a confidence approaching 1 makes the view
    near-dogmatic and a confidence approaching 0 makes it near-inert.
    """
    pi = implied_equilibrium_returns(cma, market_weights, risk_aversion)
    sigma = cma.covariance.to_numpy()
    tau_sigma = tau * sigma

    if not views:
        return BlackLittermanResult(prior_returns=pi, posterior_returns=pi.copy(), views=[])

    P, Q = _views_to_matrices(cma, views)
    base_variance = np.diag(P @ tau_sigma @ P.T)
    confidences = np.array([max(min(v.confidence, 1.0 - 1e-6), 1e-6) for v in views])
    omega = np.diag(base_variance * (1.0 / confidences - 1.0) + 1e-12)

    inv_tau_sigma = np.linalg.inv(tau_sigma)
    inv_omega = np.linalg.inv(omega)
    precision = inv_tau_sigma + P.T @ inv_omega @ P
    rhs = inv_tau_sigma @ pi.to_numpy() + P.T @ inv_omega @ Q
    posterior = np.linalg.solve(precision, rhs)

    return BlackLittermanResult(
        prior_returns=pi,
        posterior_returns=pd.Series(posterior, index=cma.names, name="posterior_return"),
        views=list(views),
    )


# --------------------------------------------------------------------------
# Policy portfolios
# --------------------------------------------------------------------------


@dataclass
class PolicyPortfolio:
    name: str
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    expected_max_drawdown: float


def expected_max_drawdown(
    mu: float,
    sigma: float,
    horizon_years: float = 10.0,
    periods_per_year: int = 12,
    n_paths: int = 2000,
    seed: int = 42,
) -> float:
    """Monte Carlo estimate of expected maximum drawdown for a GBM-like return process.

    Returns a negative fraction (e.g. -0.18 for an expected 18% peak-to-trough
    decline) averaged across simulated paths.
    """
    rng = np.random.default_rng(seed)
    n_periods = int(round(horizon_years * periods_per_year))
    period_mu = mu / periods_per_year
    period_sigma = sigma / np.sqrt(periods_per_year)
    returns = rng.normal(period_mu, period_sigma, size=(n_paths, n_periods))
    wealth = np.cumprod(1.0 + returns, axis=1)
    running_max = np.maximum.accumulate(wealth, axis=1)
    drawdowns = wealth / running_max - 1.0
    worst_per_path = drawdowns.min(axis=1)
    return float(worst_per_path.mean())


def _max_return_for_vol_cap(
    cma: CapitalMarketAssumptions,
    constraints: PortfolioConstraints,
    mu: np.ndarray,
    cov: np.ndarray,
    vol_cap: float,
) -> np.ndarray:
    n = len(mu)
    cons = [
        _equality_constraint(),
        {"type": "ineq", "fun": lambda w: vol_cap**2 - portfolio_volatility(w, cov) ** 2},
        *_inequality_constraints(cma, constraints),
    ]
    result = minimize(
        lambda w: -portfolio_return(w, mu),
        x0=_feasible_start(n, cma, constraints),
        method="SLSQP",
        bounds=_bounds(n, constraints.allow_short),
        constraints=cons,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Policy portfolio optimization failed: {result.message}")
    return result.x


def build_policy_portfolios(
    cma: CapitalMarketAssumptions,
    constraints: PortfolioConstraints | None = None,
    expected_returns: pd.Series | None = None,
    vol_targets: dict[str, float] | None = None,
    drawdown_horizon_years: float = 10.0,
    seed: int = 42,
) -> dict[str, PolicyPortfolio]:
    """Build Conservative / Moderate / Growth policy portfolios.

    Each portfolio maximizes expected return subject to a volatility cap (in
    addition to the standard institutional constraints), so risk tolerance is
    the variable that separates the three options.
    """
    constraints = constraints or PortfolioConstraints()
    mu_series = expected_returns if expected_returns is not None else cma.expected_returns
    mu = mu_series.to_numpy()
    cov = cma.covariance.to_numpy()
    cash_return = float(cma.expected_returns.get("Cash", 0.0))

    min_var_w = min_variance_portfolio(cma, constraints).to_numpy()
    max_ret_w = max_return_portfolio(cma, constraints).to_numpy()
    v_min = portfolio_volatility(min_var_w, cov)
    v_max = portfolio_volatility(max_ret_w, cov)

    default_targets = {
        "Conservative": v_min + 0.15 * (v_max - v_min),
        "Moderate": v_min + 0.50 * (v_max - v_min),
        "Growth": v_min + 0.85 * (v_max - v_min),
    }
    vol_targets = vol_targets or default_targets

    portfolios: dict[str, PolicyPortfolio] = {}
    for name, vol_cap in vol_targets.items():
        weights = _max_return_for_vol_cap(cma, constraints, mu, cov, vol_cap)
        ret = portfolio_return(weights, mu)
        vol = portfolio_volatility(weights, cov)
        sharpe = (ret - cash_return) / vol if vol > 0 else np.nan
        mdd = expected_max_drawdown(ret, vol, horizon_years=drawdown_horizon_years, seed=seed)
        portfolios[name] = PolicyPortfolio(
            name=name,
            weights=pd.Series(weights, index=cma.names, name="weight"),
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            expected_max_drawdown=mdd,
        )
    return portfolios


def compare_policy_portfolios(portfolios: dict[str, PolicyPortfolio]) -> pd.DataFrame:
    """Summarize policy portfolios' return/risk/Sharpe/drawdown side by side."""
    rows = {
        name: {
            "expected_return": p.expected_return,
            "volatility": p.volatility,
            "sharpe_ratio": p.sharpe_ratio,
            "expected_max_drawdown": p.expected_max_drawdown,
        }
        for name, p in portfolios.items()
    }
    return pd.DataFrame(rows).T
