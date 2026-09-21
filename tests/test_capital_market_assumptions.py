import numpy as np
import pytest

from endowment_ocio_toolkit.capital_market_assumptions import (
    AssetClassSpec,
    build_cma,
    is_positive_semidefinite,
)


@pytest.fixture(scope="module")
def cma():
    return build_cma()


def test_universe_size_between_8_and_10(cma):
    assert 8 <= len(cma.names) <= 10


def test_correlation_matrix_is_valid(cma):
    corr = cma.correlation.to_numpy()
    assert np.allclose(corr, corr.T), "correlation matrix must be symmetric"
    assert np.allclose(np.diag(corr), 1.0), "diagonal must be 1"
    assert np.all(corr >= -1.0 - 1e-9) and np.all(corr <= 1.0 + 1e-9), "entries must be in [-1, 1]"
    assert is_positive_semidefinite(cma.correlation), "correlation matrix must be PSD"


def test_covariance_matches_vol_and_correlation(cma):
    cov = cma.covariance
    vol = cma.volatilities
    for i in cma.names:
        for j in cma.names:
            expected = vol[i] * vol[j] * cma.correlation.loc[i, j]
            assert cov.loc[i, j] == pytest.approx(expected)


def test_return_and_volatility_sanity(cma):
    assert (cma.volatilities > 0).all()
    assert cma.volatilities.idxmin() == "Cash"
    assert cma.expected_returns["Private Equity / Venture"] > cma.expected_returns["Cash"]


def test_liquidity_tier_lookup(cma):
    assert cma.liquidity_tier("Private Equity / Venture") == "illiquid"
    assert cma.liquidity_tier("Cash") == "liquid"
    with pytest.raises(KeyError):
        cma.liquidity_tier("Not A Real Asset Class")


def test_zero_variance_loadings_are_rejected():
    bad_spec = (AssetClassSpec("A", 0.05, 0.10, "liquid", (0.0, 0.0, 0.0, 0.0), 0.0),)
    with pytest.raises(ValueError):
        build_cma(bad_spec)


def test_disclaimer_present_and_non_empty(cma):
    assert "SYNTHETIC" in cma.disclaimer.upper()
