"""End-to-end walkthrough of the endowment-ocio-toolkit modules.

Run with: python run_demo.py

DISCLAIMER: every number this script prints is generated from fabricated,
synthetic inputs (capital market assumptions, manager returns, fund cash
flows). Nothing here reflects a real client, manager, or fund, and none of it
should be used to make an investment decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from endowment_ocio_toolkit.capital_market_assumptions import build_cma, is_positive_semidefinite
from endowment_ocio_toolkit.client_reporting import (
    PositioningReport,
    SpendingStatus,
    brinson_attribution,
    build_quarterly_report,
)
from endowment_ocio_toolkit.manager_evaluation import (
    CashFlow,
    constant_growth_index,
    evaluate_liquid_manager,
    fund_multiples,
    generate_synthetic_manager_returns,
    generate_synthetic_pe_cash_flows,
    irr,
    kaplan_schoar_pme,
)
from endowment_ocio_toolkit.portfolio_rebalancing import RebalancingPolicy, compare_rebalancing_policies
from endowment_ocio_toolkit.spending_policy_model import SpendingPolicyParams, simulate_endowment, summarize_simulation
from endowment_ocio_toolkit.strategic_asset_allocation import (
    HouseView,
    PortfolioConstraints,
    black_litterman_posterior,
    build_policy_portfolios,
    compare_policy_portfolios,
)

pd.set_option("display.width", 140)
pd.set_option("display.precision", 4)


def section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def main() -> None:
    section("0. DISCLAIMER")
    print(
        "All figures below come from synthetic, fabricated data for demonstration\n"
        "purposes only. No real client, manager, or fund is represented."
    )

    # ------------------------------------------------------------------
    section("1. Capital Market Assumptions")
    cma = build_cma()
    print(pd.DataFrame({"expected_return": cma.expected_returns, "volatility": cma.volatilities}))
    print(f"\nCorrelation matrix is positive semi-definite: {is_positive_semidefinite(cma.correlation)}")

    # ------------------------------------------------------------------
    section("2. Strategic Asset Allocation: Policy Portfolios")
    constraints = PortfolioConstraints(max_private_equity=0.15, max_illiquid_total=0.35, min_liquidity_buffer=0.03)
    policy_portfolios = build_policy_portfolios(cma, constraints)
    print(compare_policy_portfolios(policy_portfolios))
    print("\nModerate portfolio weights:")
    print(policy_portfolios["Moderate"].weights.round(4))

    section("2b. Black-Litterman: a house view on Emerging Markets Equity")
    market_weights = pd.Series(1.0 / len(cma.names), index=cma.names)
    house_view = HouseView(
        assets=["Emerging Markets Equity", "Core Fixed Income"],
        view_weights=[1.0, -1.0],
        magnitude=0.08,
        confidence=0.7,
    )
    bl_result = black_litterman_posterior(cma, market_weights, views=[house_view])
    print(pd.DataFrame({"prior": bl_result.prior_returns, "posterior": bl_result.posterior_returns}))

    # ------------------------------------------------------------------
    section("3. Spending Policy: 20-Year Monte Carlo")
    spending_params = SpendingPolicyParams(annual_spend_rate=0.045)
    for name, portfolio in policy_portfolios.items():
        result = simulate_endowment(
            portfolio.expected_return,
            portfolio.volatility,
            params=spending_params,
            n_paths=2000,
            horizon_years=20,
            seed=42,
        )
        summary = summarize_simulation(result, initial_value=100.0)
        print(f"\n{name} policy portfolio:")
        for key, value in summary.items():
            print(f"  {key}: {value:.4f}")

    # ------------------------------------------------------------------
    section("4. Manager Evaluation: Liquid Strategy")
    manager_returns, benchmark_returns = generate_synthetic_manager_returns(n_periods=60)
    liquid_report = evaluate_liquid_manager(manager_returns, benchmark_returns)
    print(f"Tracking error: {liquid_report.tracking_error:.4f}")
    print(f"Information ratio: {liquid_report.information_ratio:.4f}")
    print(f"Up capture: {liquid_report.up_capture:.2f}%  Down capture: {liquid_report.down_capture:.2f}%")
    print("Rolling Sharpe (last 5 obs):")
    print(liquid_report.rolling_sharpe.tail())

    section("4b. Manager Evaluation: Private Equity Fund")
    pe_cash_flows, pe_nav = generate_synthetic_pe_cash_flows(fund_size=100.0, fund_life_years=10.0, moic_target=2.3)
    all_flows = pe_cash_flows + ([CashFlow(10.0, pe_nav)] if pe_nav > 0 else [])
    fund_irr = irr(all_flows)
    multiples = fund_multiples(pe_cash_flows, current_nav=pe_nav)
    index = constant_growth_index(annual_return=0.08)
    pme = kaplan_schoar_pme(pe_cash_flows, current_nav=pe_nav, index_level_at=index, valuation_period=10.0)
    print(f"IRR: {fund_irr:.4f}")
    print(f"TVPI: {multiples.tvpi:.2f}  DPI: {multiples.dpi:.2f}  RVPI: {multiples.rvpi:.2f}")
    print(f"KS-PME vs. an 8%/yr public index: {pme:.2f}")

    # ------------------------------------------------------------------
    section("5. Portfolio Rebalancing")
    moderate_weights = policy_portfolios["Moderate"].weights
    illiquid_assets = ["Private Equity / Venture", "Hedge Funds - Diversifying", "Hedge Funds - Directional"]
    n_quarters = 40
    rng = np.random.default_rng(7)
    quarterly_mu = cma.expected_returns / 4.0
    quarterly_cov = cma.covariance / 4.0
    simulated_returns = pd.DataFrame(
        rng.multivariate_normal(quarterly_mu.to_numpy(), quarterly_cov.to_numpy(), size=n_quarters),
        columns=cma.names,
    )
    comparison = compare_rebalancing_policies(
        moderate_weights,
        simulated_returns,
        illiquid_assets=illiquid_assets,
        policy=RebalancingPolicy(band=0.03),
    )
    print(f"Rebalanced tracking error to target: {comparison.rebalanced_tracking_error:.4f}")
    print(f"No-rebalance tracking error to target: {comparison.no_rebalance_tracking_error:.4f}")

    # ------------------------------------------------------------------
    section("6. Client Reporting: Quarterly Report")
    actual_weights = comparison.rebalanced_weights.iloc[-1]
    actual_weights.index = cma.names
    positioning = PositioningReport(target_weights=moderate_weights, actual_weights=actual_weights)

    # A simple single-period attribution using the last simulated quarter's returns
    # as "portfolio" segment returns and the policy return as the benchmark segment return.
    last_quarter_returns = simulated_returns.iloc[-1]
    benchmark_segment_returns = quarterly_mu
    attribution = brinson_attribution(actual_weights, moderate_weights, last_quarter_returns, benchmark_segment_returns)
    print(attribution.detail.round(5))
    print(f"\nReconciles (allocation + selection + interaction == total active return): {attribution.reconciles()}")

    spending_status = SpendingStatus(
        annual_spend_rate=spending_params.annual_spend_rate,
        trailing_average_market_value=98.5,
        current_market_value=101.2,
        quarterly_spend_amount=1.14,
    )
    report = build_quarterly_report("Q2 2026 (synthetic)", positioning, attribution, spending_status)
    print("\n" + report.executive_summary)


if __name__ == "__main__":
    main()
