import numpy as np
import pandas as pd
import pytest

from endowment_ocio_toolkit.client_reporting import (
    PositioningReport,
    SpendingStatus,
    brinson_attribution,
    build_quarterly_report,
    generate_executive_summary,
)

NAMES = ["US Equity", "Core Fixed Income", "Private Equity / Venture"]


@pytest.fixture
def positioning():
    target = pd.Series([0.5, 0.3, 0.2], index=NAMES)
    actual = pd.Series([0.55, 0.25, 0.20], index=NAMES)
    return PositioningReport(target_weights=target, actual_weights=actual)


@pytest.fixture
def attribution_inputs():
    weights_p = pd.Series([0.55, 0.25, 0.20], index=NAMES)
    weights_b = pd.Series([0.5, 0.3, 0.2], index=NAMES)
    returns_p = pd.Series([0.06, 0.02, 0.09], index=NAMES)
    returns_b = pd.Series([0.05, 0.015, 0.08], index=NAMES)
    return weights_p, weights_b, returns_p, returns_b


def test_positioning_active_weights_and_frame(positioning):
    active = positioning.active_weights
    assert active["US Equity"] == pytest.approx(0.05)
    assert active["Core Fixed Income"] == pytest.approx(-0.05)
    assert active["Private Equity / Venture"] == pytest.approx(0.0)
    assert set(positioning.as_frame().columns) == {"target_weight", "actual_weight", "active_weight"}


def test_brinson_reconciles_and_matches_return_difference(attribution_inputs):
    wp, wb, rp, rb = attribution_inputs
    attribution = brinson_attribution(wp, wb, rp, rb)
    assert attribution.reconciles()
    assert attribution.total_active_return == pytest.approx(
        attribution.portfolio_return - attribution.benchmark_return
    )


def test_brinson_selection_and_allocation_effects_isolate_correctly():
    names = ["A", "B"]
    # Same returns everywhere: selection and interaction must vanish.
    wp1, wb1 = pd.Series([0.6, 0.4], index=names), pd.Series([0.5, 0.5], index=names)
    r = pd.Series([0.05, 0.03], index=names)
    same_returns = brinson_attribution(wp1, wb1, r, r)
    assert same_returns.selection_effect == pytest.approx(0.0, abs=1e-12)
    assert same_returns.interaction_effect == pytest.approx(0.0, abs=1e-12)

    # Same weights everywhere: allocation and interaction must vanish.
    w = pd.Series([0.5, 0.5], index=names)
    rp2 = pd.Series([0.05, 0.03], index=names)
    rb2 = pd.Series([0.04, 0.02], index=names)
    same_weights = brinson_attribution(w, w, rp2, rb2)
    assert same_weights.allocation_effect == pytest.approx(0.0, abs=1e-12)
    assert same_weights.interaction_effect == pytest.approx(0.0, abs=1e-12)


def test_brinson_reconciles_on_random_inputs():
    rng = np.random.default_rng(0)
    names = [f"asset_{i}" for i in range(6)]
    for _ in range(20):
        wp = pd.Series(rng.dirichlet(np.ones(6)), index=names)
        wb = pd.Series(rng.dirichlet(np.ones(6)), index=names)
        rp = pd.Series(rng.normal(0.05, 0.1, 6), index=names)
        rb = pd.Series(rng.normal(0.05, 0.1, 6), index=names)
        assert brinson_attribution(wp, wb, rp, rb).reconciles(tol=1e-9)


def test_spending_status_effective_annualized_rate():
    status = SpendingStatus(0.045, 98.0, 100.0, 1.1)
    assert status.effective_annualized_spend_pct_of_current_value == pytest.approx(1.1 * 4 / 100.0)


def test_executive_summary_contains_key_figures(positioning, attribution_inputs):
    wp, wb, rp, rb = attribution_inputs
    attribution = brinson_attribution(wp, wb, rp, rb)
    status = SpendingStatus(0.045, 98.0, 100.0, 1.1)
    summary = generate_executive_summary(positioning, attribution, status, as_of="Q2 2026")
    assert "Q2 2026" in summary
    assert "US Equity" in summary
    assert "SYNTHETIC" in summary.upper()


def test_build_quarterly_report_bundles_all_sections(positioning, attribution_inputs):
    wp, wb, rp, rb = attribution_inputs
    attribution = brinson_attribution(wp, wb, rp, rb)
    status = SpendingStatus(0.045, 98.0, 100.0, 1.1)
    report = build_quarterly_report("Q2 2026", positioning, attribution, status)
    assert report.as_of == "Q2 2026"
    assert report.positioning is positioning
    assert report.attribution is attribution
    assert report.spending_status is status
    assert len(report.executive_summary) > 0
