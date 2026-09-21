"""Policy portfolio drift, rebalancing bands, and cash-flow-based rebalancing.

DISCLAIMER: Return paths in this module are synthetic and for demonstration
only.

Private-market allocations (private equity/venture, and by convention here
also the semi-liquid hedge fund and real-assets sleeves) cannot be traded on
demand: an OCIO cannot sell down an overweight PE position to fund a
rebalance the way it can with public equities or bonds. This module models
that constraint explicitly: only the "liquid sleeve" is rebalanced, and
illiquid weights are left to drift with markets (and, optionally, with
scheduled capital calls/distributions).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DISCLAIMER = (
    "SYNTHETIC DATA: return paths and drift simulations in this module are "
    "fabricated for demonstration only."
)


@dataclass(frozen=True)
class RebalancingPolicy:
    """A bands-based rebalancing policy applied to the liquid sleeve only.

    Attributes:
        band: Absolute weight deviation from target (e.g. 0.03 = 300 bps)
            that triggers a rebalance of the liquid sleeve back to target.
        rebalance_check_frequency: Rebalancing is evaluated every N periods
            (e.g. quarterly if period = quarter).
    """

    band: float = 0.03
    rebalance_check_frequency: int = 1


def simulate_drift(
    target_weights: pd.Series,
    asset_returns: pd.DataFrame,
    illiquid_assets: list[str],
    policy: RebalancingPolicy | None = None,
    external_cash_flows: pd.Series | None = None,
) -> dict[str, pd.DataFrame]:
    """Simulate portfolio weight drift with and without a rebalancing policy.

    At each period, each asset class earns its own return. Illiquid asset
    weights are never traded (they simply drift, renormalized only because
    total portfolio value changed). The liquid sleeve is rebalanced back to
    its target *relative* proportions whenever any liquid asset's absolute
    weight deviation from target exceeds ``policy.band``. If
    ``external_cash_flows`` is given (positive = net contribution, negative =
    net withdrawal, as a fraction of prior period value), fresh cash is
    invested into (or drawn from) the liquid sleeve in proportion to how far
    each liquid asset is below (or above) its target -- "cash flow-based"
    rebalancing -- before the band check.

    Args:
        target_weights: Target policy weights, indexed by asset class.
        asset_returns: Periodic returns, shape (n_periods, n_assets), columns
            matching ``target_weights.index``.
        illiquid_assets: Asset class names that cannot be traded on demand.
        policy: Rebalancing policy; defaults applied if omitted.
        external_cash_flows: Optional per-period net external cash flow, as a
            fraction of the prior period's total portfolio value.

    Returns:
        Dict with keys ``"rebalanced"`` and ``"no_rebalance"``, each mapping
        to a DataFrame of weights over time (shape n_periods+1 x n_assets).
    """
    policy = policy or RebalancingPolicy()
    names = list(target_weights.index)
    liquid_assets = [n for n in names if n not in illiquid_assets]
    target = target_weights.reindex(names).to_numpy()
    liquid_mask = np.array([n in liquid_assets for n in names])
    returns = asset_returns.reindex(columns=names).to_numpy()
    n_periods = returns.shape[0]

    def _run(rebalance: bool, use_cash_flows: bool) -> pd.DataFrame:
        weights = np.empty((n_periods + 1, len(names)))
        weights[0] = target.copy()
        w = target.copy()
        for t in range(n_periods):
            grown = w * (1.0 + returns[t])
            w = grown / grown.sum()

            if use_cash_flows and external_cash_flows is not None:
                flow = float(external_cash_flows.iloc[t]) if t < len(external_cash_flows) else 0.0
                if flow != 0.0 and liquid_mask.any():
                    gap = (target - w) * liquid_mask
                    if flow > 0 and gap[gap > 0].sum() > 0:
                        allocation = np.where(gap > 0, gap, 0.0)
                        allocation = allocation / allocation.sum()
                        w = _apply_contribution(w, allocation, flow)
                    elif flow < 0 and liquid_mask.sum() > 0:
                        w = _apply_withdrawal(w, liquid_mask, -flow)

            if rebalance and (t + 1) % policy.rebalance_check_frequency == 0:
                deviation = (w - target) * liquid_mask
                if np.any(np.abs(deviation) > policy.band):
                    illiquid_total = w[~liquid_mask].sum()
                    liquid_target_total = target[liquid_mask].sum()
                    if liquid_target_total > 0:
                        w[liquid_mask] = target[liquid_mask] / target[liquid_mask].sum() * (1 - illiquid_total)
            weights[t + 1] = w
        return pd.DataFrame(weights, columns=names)

    return {
        "rebalanced": _run(rebalance=True, use_cash_flows=external_cash_flows is not None),
        "no_rebalance": _run(rebalance=False, use_cash_flows=False),
    }


def _apply_contribution(weights: np.ndarray, allocation: np.ndarray, flow_fraction: float) -> np.ndarray:
    """Invest new external cash (a fraction ``flow_fraction`` of portfolio value) per ``allocation``.

    ``allocation`` sums to 1 across the assets receiving the new cash (e.g.
    the most underweight liquid asset classes). Since ``weights`` sum to 1
    (fractions of pre-flow portfolio value), the post-flow weights are
    ``(weights + allocation * flow_fraction) / (1 + flow_fraction)``.
    """
    new_total = 1.0 + flow_fraction
    return (weights + allocation * flow_fraction) / new_total


def _apply_withdrawal(weights: np.ndarray, liquid_mask: np.ndarray, flow_fraction: float) -> np.ndarray:
    """Withdraw cash (a fraction ``flow_fraction`` of portfolio value) pro rata from the liquid sleeve."""
    new_total = 1.0 - flow_fraction
    dollar_weights = weights.copy()
    liquid_total = dollar_weights[liquid_mask].sum()
    if liquid_total > 0:
        dollar_weights[liquid_mask] -= dollar_weights[liquid_mask] / liquid_total * flow_fraction
    return dollar_weights / new_total


def portfolio_returns_from_weights(weights: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.Series:
    """Realized portfolio return each period given start-of-period weights and asset returns."""
    names = list(asset_returns.columns)
    w = weights[names].to_numpy()[:-1]  # start-of-period weights
    r = asset_returns[names].to_numpy()
    return pd.Series((w * r).sum(axis=1), name="portfolio_return")


def tracking_error_to_target(
    weights: pd.DataFrame,
    target_weights: pd.Series,
    asset_returns: pd.DataFrame,
    periods_per_year: int = 4,
) -> float:
    """Annualized tracking error of a drifting/rebalanced portfolio versus a fixed-weight benchmark."""
    names = list(asset_returns.columns)
    port_returns = portfolio_returns_from_weights(weights, asset_returns)
    target = target_weights.reindex(names).to_numpy()
    bench_returns = (asset_returns[names].to_numpy() * target).sum(axis=1)
    active = port_returns.to_numpy() - bench_returns
    return float(np.std(active, ddof=1) * np.sqrt(periods_per_year))


@dataclass
class RebalancingComparison:
    rebalanced_weights: pd.DataFrame
    no_rebalance_weights: pd.DataFrame
    rebalanced_tracking_error: float
    no_rebalance_tracking_error: float


def compare_rebalancing_policies(
    target_weights: pd.Series,
    asset_returns: pd.DataFrame,
    illiquid_assets: list[str],
    policy: RebalancingPolicy | None = None,
    external_cash_flows: pd.Series | None = None,
    periods_per_year: int = 4,
) -> RebalancingComparison:
    """Simulate drift and compute tracking error for rebalanced vs. buy-and-hold portfolios."""
    result = simulate_drift(target_weights, asset_returns, illiquid_assets, policy, external_cash_flows)
    return RebalancingComparison(
        rebalanced_weights=result["rebalanced"],
        no_rebalance_weights=result["no_rebalance"],
        rebalanced_tracking_error=tracking_error_to_target(
            result["rebalanced"], target_weights, asset_returns, periods_per_year
        ),
        no_rebalance_tracking_error=tracking_error_to_target(
            result["no_rebalance"], target_weights, asset_returns, periods_per_year
        ),
    )
