"""Smoothed endowment spending policy and 20-year Monte Carlo projection.

DISCLAIMER: All simulated market values, spending amounts, and probabilities
in this module are synthetic and for demonstration only.

The spending rule modeled here is the common "smoothed" or "moving-average"
policy used by many endowments and foundations: each quarter, spending equals
``spend_rate / 4`` times the trailing 12-quarter (3-year) average market value.
Smoothing on a multi-year trailing average dampens the effect of short-term
market swings on the annual spending draw that funds the institution's budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DISCLAIMER = (
    "SYNTHETIC DATA: this Monte Carlo projection uses fabricated capital "
    "market assumptions and is for demonstration only, not financial advice."
)


@dataclass(frozen=True)
class SpendingPolicyParams:
    """Parameters of a smoothed endowment spending rule."""

    annual_spend_rate: float = 0.045
    smoothing_quarters: int = 12
    inflation_mean: float = 0.025
    inflation_vol: float = 0.01


def trailing_average_market_value(market_values: np.ndarray, smoothing_quarters: int) -> np.ndarray:
    """Trailing (inclusive) moving average of market value at each quarter.

    For quarters before ``smoothing_quarters`` of history exist, the average
    is taken over however much history is available (matching common
    practice for a newly established smoothing window).

    Args:
        market_values: 1-D array of market values, index 0 = quarter 0 (start).
        smoothing_quarters: Number of trailing quarters to average over.

    Returns:
        Array the same length as ``market_values`` with the trailing average
        as of (and including) each quarter.
    """
    n = len(market_values)
    averages = np.empty(n)
    for t in range(n):
        start = max(0, t - smoothing_quarters + 1)
        averages[t] = market_values[start : t + 1].mean()
    return averages


def quarterly_spend(trailing_average: float, annual_spend_rate: float) -> float:
    """Quarterly dollar spend implied by the smoothed rule."""
    return annual_spend_rate / 4.0 * trailing_average


@dataclass
class SimulationResult:
    """Container for one Monte Carlo spending-policy simulation."""

    market_value_paths: np.ndarray  # (n_paths, n_quarters + 1), nominal
    real_value_paths: np.ndarray  # inflation-adjusted, same shape
    spending_paths: np.ndarray  # (n_paths, n_quarters), nominal quarterly spend
    real_spending_paths: np.ndarray  # inflation-adjusted quarterly spend
    horizon_years: int

    @property
    def n_paths(self) -> int:
        return self.market_value_paths.shape[0]

    def probability_real_value_preserved(self, initial_value: float) -> float:
        """Share of paths whose ending real market value is >= the starting value."""
        ending_real = self.real_value_paths[:, -1]
        return float(np.mean(ending_real >= initial_value))

    def annual_real_spending(self) -> np.ndarray:
        """Real spending aggregated to annual totals, shape (n_paths, horizon_years)."""
        n_quarters = self.real_spending_paths.shape[1]
        n_years = n_quarters // 4
        trimmed = self.real_spending_paths[:, : n_years * 4]
        return trimmed.reshape(self.n_paths, n_years, 4).sum(axis=2)

    def spending_volatility(self) -> float:
        """Cross-path average of each path's year-over-year real spending volatility.

        For each simulated path, this is the standard deviation of annual
        percentage changes in real spending; the reported figure averages
        that across all paths.
        """
        annual = self.annual_real_spending()
        pct_changes = np.diff(annual, axis=1) / annual[:, :-1]
        per_path_vol = pct_changes.std(axis=1, ddof=1)
        return float(per_path_vol.mean())

    def median_ending_real_value(self) -> float:
        return float(np.median(self.real_value_paths[:, -1]))


def simulate_endowment(
    expected_return: float,
    volatility: float,
    params: SpendingPolicyParams | None = None,
    initial_value: float = 100.0,
    horizon_years: int = 20,
    n_paths: int = 5000,
    seed: int = 7,
) -> SimulationResult:
    """Run a 20-year Monte Carlo simulation of the smoothed spending policy.

    Each quarter: the endowment spends ``annual_spend_rate / 4`` times the
    trailing moving-average market value (computed on nominal values, as is
    standard practice), then the remaining balance compounds at a randomly
    drawn quarterly portfolio return. A separate stochastic inflation path
    deflates nominal values and spending to "real" (today's dollar) terms.

    Args:
        expected_return: Annualized arithmetic expected return of the policy
            portfolio (e.g. from ``strategic_asset_allocation``).
        volatility: Annualized volatility of the policy portfolio.
        params: Spending policy parameters (rate, smoothing window, inflation).
        initial_value: Starting nominal market value.
        horizon_years: Simulation horizon in years.
        n_paths: Number of Monte Carlo paths.
        seed: Seed for reproducibility.

    Returns:
        A :class:`SimulationResult` with nominal/real market value and
        spending paths.
    """
    params = params or SpendingPolicyParams()
    rng = np.random.default_rng(seed)
    n_quarters = horizon_years * 4

    q_mu = expected_return / 4.0
    q_sigma = volatility / 2.0  # vol scales with sqrt(4) = 2
    returns = rng.normal(q_mu, q_sigma, size=(n_paths, n_quarters))

    infl_q_mu = params.inflation_mean / 4.0
    infl_q_sigma = params.inflation_vol / 2.0
    inflation = rng.normal(infl_q_mu, infl_q_sigma, size=(n_paths, n_quarters))
    cumulative_inflation = np.cumprod(1.0 + inflation, axis=1)
    cumulative_inflation = np.concatenate(
        [np.ones((n_paths, 1)), cumulative_inflation], axis=1
    )  # quarter 0 = 1.0

    market_values = np.empty((n_paths, n_quarters + 1))
    spending = np.empty((n_paths, n_quarters))
    market_values[:, 0] = initial_value

    for path in range(n_paths):
        mv = market_values[path]
        for t in range(n_quarters):
            trailing_avg = mv[max(0, t - params.smoothing_quarters + 1) : t + 1].mean()
            spend = quarterly_spend(trailing_avg, params.annual_spend_rate)
            spend = min(spend, mv[t])  # cannot spend more than the balance
            spending[path, t] = spend
            mv[t + 1] = (mv[t] - spend) * (1.0 + returns[path, t])

    real_market_values = market_values / cumulative_inflation
    real_spending = spending / cumulative_inflation[:, 1:]

    return SimulationResult(
        market_value_paths=market_values,
        real_value_paths=real_market_values,
        spending_paths=spending,
        real_spending_paths=real_spending,
        horizon_years=horizon_years,
    )


def summarize_simulation(result: SimulationResult, initial_value: float) -> dict[str, float]:
    """Headline statistics reported to clients/committees for a spending simulation."""
    return {
        "probability_real_value_preserved": result.probability_real_value_preserved(initial_value),
        "median_ending_real_value": result.median_ending_real_value(),
        "spending_volatility": result.spending_volatility(),
        "median_ending_real_value_ratio": result.median_ending_real_value() / initial_value,
    }
