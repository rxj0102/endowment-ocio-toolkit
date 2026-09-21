"""Client-facing quarterly report assembly: positioning, attribution, spending status.

DISCLAIMER: All figures produced or consumed by this module in the toolkit's
examples/tests are synthetic and for demonstration only.

Performance attribution uses the classic Brinson-Hood-Beebower (BHB) model,
which decomposes total active return (portfolio return minus benchmark/policy
return) into three additive effects that reconcile exactly:

    allocation_i  = (w_p,i - w_b,i) * r_b,i
    selection_i   = w_b,i * (r_p,i - r_b,i)
    interaction_i = (w_p,i - w_b,i) * (r_p,i - r_b,i)

    sum_i (allocation_i + selection_i + interaction_i) = R_p - R_b

Allocation captures the value added (or lost) by tilting sector/asset-class
weights away from the policy benchmark; selection captures the value added by
manager/security choice within each sleeve; interaction captures the
cross-term between the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DISCLAIMER = (
    "SYNTHETIC DATA: this report is generated from fabricated positions and "
    "returns for demonstration only and does not describe any real client "
    "account."
)


@dataclass
class PositioningReport:
    """Policy portfolio positioning vs. targets."""

    target_weights: pd.Series
    actual_weights: pd.Series

    @property
    def active_weights(self) -> pd.Series:
        return (self.actual_weights - self.target_weights).rename("active_weight")

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "target_weight": self.target_weights,
                "actual_weight": self.actual_weights,
                "active_weight": self.active_weights,
            }
        )


@dataclass
class BrinsonAttribution:
    """Per-asset-class and total Brinson-Hood-Beebower attribution results."""

    detail: pd.DataFrame  # columns: allocation, selection, interaction, total_effect
    portfolio_return: float
    benchmark_return: float

    @property
    def total_active_return(self) -> float:
        return self.portfolio_return - self.benchmark_return

    @property
    def allocation_effect(self) -> float:
        return float(self.detail["allocation"].sum())

    @property
    def selection_effect(self) -> float:
        return float(self.detail["selection"].sum())

    @property
    def interaction_effect(self) -> float:
        return float(self.detail["interaction"].sum())

    def reconciles(self, tol: float = 1e-9) -> bool:
        total = self.allocation_effect + self.selection_effect + self.interaction_effect
        return bool(abs(total - self.total_active_return) < tol)


def brinson_attribution(
    portfolio_weights: pd.Series,
    benchmark_weights: pd.Series,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> BrinsonAttribution:
    """Compute a Brinson-Hood-Beebower single-period attribution.

    All four inputs must share the same asset-class index.
    """
    names = list(portfolio_weights.index)
    wp = portfolio_weights.reindex(names)
    wb = benchmark_weights.reindex(names)
    rp = portfolio_returns.reindex(names)
    rb = benchmark_returns.reindex(names)

    active_weight = wp - wb
    active_return = rp - rb

    allocation = active_weight * rb
    selection = wb * active_return
    interaction = active_weight * active_return
    total_effect = allocation + selection + interaction

    detail = pd.DataFrame(
        {
            "allocation": allocation,
            "selection": selection,
            "interaction": interaction,
            "total_effect": total_effect,
        }
    )

    portfolio_return = float((wp * rp).sum())
    benchmark_return = float((wb * rb).sum())

    return BrinsonAttribution(detail=detail, portfolio_return=portfolio_return, benchmark_return=benchmark_return)


@dataclass
class SpendingStatus:
    """Spending policy status section of the quarterly report."""

    annual_spend_rate: float
    trailing_average_market_value: float
    current_market_value: float
    quarterly_spend_amount: float

    @property
    def effective_annualized_spend_pct_of_current_value(self) -> float:
        if self.current_market_value == 0:
            return float("nan")
        return (self.quarterly_spend_amount * 4.0) / self.current_market_value


@dataclass
class QuarterlyReport:
    as_of: str
    positioning: PositioningReport
    attribution: BrinsonAttribution
    spending_status: SpendingStatus
    executive_summary: str


def _fmt_pct(x: float, decimals: int = 2) -> str:
    return f"{x * 100:.{decimals}f}%"


def generate_executive_summary(
    positioning: PositioningReport,
    attribution: BrinsonAttribution,
    spending_status: SpendingStatus,
    as_of: str,
) -> str:
    """Auto-generate a short plain-language executive summary from report components."""
    active = positioning.active_weights
    largest_overweight = active.idxmax()
    largest_underweight = active.idxmin()

    lines = [
        f"Quarterly Portfolio Summary -- {as_of}",
        "",
        (
            f"The policy portfolio returned {_fmt_pct(attribution.portfolio_return)} versus "
            f"{_fmt_pct(attribution.benchmark_return)} for the policy benchmark, a total active "
            f"return of {_fmt_pct(attribution.total_active_return)}."
        ),
        (
            f"Of that active return, asset allocation contributed "
            f"{_fmt_pct(attribution.allocation_effect)}, manager/security selection contributed "
            f"{_fmt_pct(attribution.selection_effect)}, and the interaction of the two contributed "
            f"{_fmt_pct(attribution.interaction_effect)}."
        ),
        (
            f"The largest active positioning tilt is a "
            f"{'overweight' if active[largest_overweight] >= 0 else 'underweight'} of "
            f"{_fmt_pct(abs(active[largest_overweight]))} in {largest_overweight}, while the "
            f"largest underweight is {_fmt_pct(abs(active[largest_underweight]))} in {largest_underweight}."
        ),
        (
            f"Spending policy: the trailing smoothed market value average of "
            f"{spending_status.trailing_average_market_value:,.1f} supported a quarterly spend of "
            f"{spending_status.quarterly_spend_amount:,.1f}, equivalent to an effective annualized "
            f"draw of {_fmt_pct(spending_status.effective_annualized_spend_pct_of_current_value)} on "
            f"the current market value of {spending_status.current_market_value:,.1f}."
        ),
        "",
        DISCLAIMER,
    ]
    return "\n".join(lines)


def build_quarterly_report(
    as_of: str,
    positioning: PositioningReport,
    attribution: BrinsonAttribution,
    spending_status: SpendingStatus,
) -> QuarterlyReport:
    """Assemble the full quarterly report, including the auto-generated executive summary."""
    summary = generate_executive_summary(positioning, attribution, spending_status, as_of)
    return QuarterlyReport(
        as_of=as_of,
        positioning=positioning,
        attribution=attribution,
        spending_status=spending_status,
        executive_summary=summary,
    )
