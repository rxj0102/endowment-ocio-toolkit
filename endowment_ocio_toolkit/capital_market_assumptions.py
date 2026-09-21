"""Synthetic 10-year capital market assumptions (CMA) for an endowment policy portfolio.

DISCLAIMER: All figures in this module are fabricated for demonstration purposes
only. They do not represent the views, research, or capital market assumptions
of any real firm, and must never be used for actual investment decisions.

The correlation matrix is built from a small latent-factor model (equity beta,
duration/rates, illiquidity, and diversifying-alternatives factors). Because any
matrix of the form L @ L.T is a Gram matrix, it is guaranteed to be positive
semi-definite; rescaling it to unit diagonal (a congruence transform by a
positive diagonal matrix) preserves that property, so the resulting correlation
matrix is valid by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DISCLAIMER = (
    "SYNTHETIC DATA: capital market assumptions in this module are illustrative "
    "and fabricated for a portfolio-analytics demonstration. They are not the "
    "research output of any real institution and must not be used to make "
    "investment decisions."
)

#: Liquidity tiers used elsewhere in the toolkit (rebalancing, constraints).
LIQUID = "liquid"
SEMI_LIQUID = "semi_liquid"
ILLIQUID = "illiquid"


@dataclass(frozen=True)
class AssetClassSpec:
    """Static description of one asset class in the CMA set."""

    name: str
    expected_return: float
    volatility: float
    liquidity_tier: str
    factor_loadings: tuple[float, float, float, float]
    idiosyncratic_loading: float = 0.0


#: Factor order: (equity_beta, rates_duration, illiquidity, diversifying_alts).
#: Each spec also carries an ``idiosyncratic_loading`` -- a diversifiable,
#: asset-specific variance component not explained by the common factors.
#: Without it, asset classes that share similar factor exposure (e.g. US and
#: EM equity both loading heavily on equity beta) would be numerically
#: near-collinear, which is unrealistic and destabilizes matrix inversion
#: (e.g. in Black-Litterman). Idiosyncratic variance is added as an extra
#: diagonal block in the underlying Gram matrix, which keeps the
#: construction positive semi-definite by construction while pulling
#: cross-correlations down to realistic levels.
_ASSET_CLASSES: tuple[AssetClassSpec, ...] = (
    AssetClassSpec("US Equity", 0.065, 0.17, LIQUID, (0.85, -0.05, 0.05, 0.10), 0.35),
    AssetClassSpec("Non-US Developed Equity", 0.070, 0.18, LIQUID, (0.80, -0.05, 0.05, 0.10), 0.40),
    AssetClassSpec("Emerging Markets Equity", 0.078, 0.24, LIQUID, (0.75, -0.05, 0.10, 0.15), 0.55),
    AssetClassSpec("Core Fixed Income", 0.045, 0.06, LIQUID, (-0.10, 0.85, 0.00, 0.10), 0.30),
    AssetClassSpec("Private Equity / Venture", 0.095, 0.24, ILLIQUID, (0.55, 0.00, 0.70, 0.10), 0.45),
    AssetClassSpec("Hedge Funds - Diversifying", 0.050, 0.07, SEMI_LIQUID, (0.20, 0.05, 0.05, 0.85), 0.35),
    AssetClassSpec("Hedge Funds - Directional", 0.062, 0.10, SEMI_LIQUID, (0.55, 0.00, 0.05, 0.55), 0.40),
    AssetClassSpec("Real Assets / Real Estate", 0.060, 0.14, SEMI_LIQUID, (0.45, 0.15, 0.35, 0.20), 0.45),
    AssetClassSpec("Cash", 0.030, 0.01, LIQUID, (0.00, 0.10, 0.00, 0.00), 0.20),
)


@dataclass
class CapitalMarketAssumptions:
    """Bundled 10-year CMA set: expected returns, vols, and a correlation matrix."""

    asset_classes: tuple[AssetClassSpec, ...]
    expected_returns: pd.Series
    volatilities: pd.Series
    correlation: pd.DataFrame
    disclaimer: str = field(default=DISCLAIMER)

    @property
    def names(self) -> list[str]:
        return list(self.expected_returns.index)

    @property
    def covariance(self) -> pd.DataFrame:
        """Covariance matrix implied by volatilities and the correlation matrix."""
        vol = self.volatilities
        cov = self.correlation.mul(vol, axis=0).mul(vol, axis=1)
        return cov

    def liquidity_tier(self, name: str) -> str:
        for spec in self.asset_classes:
            if spec.name == name:
                return spec.liquidity_tier
        raise KeyError(f"Unknown asset class: {name}")

    def tier_members(self, tier: str) -> list[str]:
        return [spec.name for spec in self.asset_classes if spec.liquidity_tier == tier]


def _build_correlation_matrix(specs: tuple[AssetClassSpec, ...]) -> pd.DataFrame:
    names = [s.name for s in specs]
    loadings = np.array([s.factor_loadings for s in specs])
    idio = np.array([s.idiosyncratic_loading for s in specs])
    gram = loadings @ loadings.T + np.diag(idio**2)
    diag = np.diag(gram)
    if np.any(diag <= 0):
        raise ValueError("Factor loadings must give every asset class a positive self-variance.")
    scale = np.sqrt(np.outer(diag, diag))
    corr = gram / scale
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=names, columns=names)


def build_cma(asset_classes: tuple[AssetClassSpec, ...] | None = None) -> CapitalMarketAssumptions:
    """Construct the synthetic 10-year CMA set.

    Args:
        asset_classes: Optional override of the default 9-asset-class universe,
            e.g. for sensitivity testing.

    Returns:
        A :class:`CapitalMarketAssumptions` bundle with expected returns,
        volatilities, and a validated correlation matrix.
    """
    specs = asset_classes if asset_classes is not None else _ASSET_CLASSES
    names = [s.name for s in specs]
    expected_returns = pd.Series([s.expected_return for s in specs], index=names, name="expected_return")
    volatilities = pd.Series([s.volatility for s in specs], index=names, name="volatility")
    correlation = _build_correlation_matrix(specs)
    return CapitalMarketAssumptions(
        asset_classes=specs,
        expected_returns=expected_returns,
        volatilities=volatilities,
        correlation=correlation,
    )


def is_positive_semidefinite(matrix: pd.DataFrame | np.ndarray, tol: float = 1e-8) -> bool:
    """Check whether a symmetric matrix is positive semi-definite within tolerance."""
    values = matrix.to_numpy() if isinstance(matrix, pd.DataFrame) else np.asarray(matrix)
    eigenvalues = np.linalg.eigvalsh((values + values.T) / 2.0)
    return bool(np.all(eigenvalues >= -tol))
