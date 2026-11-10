# endowment-ocio-toolkit

[![CI](https://github.com/rxj0102/endowment-ocio-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/endowment-ocio-toolkit/actions/workflows/ci.yml)

> **DISCLAIMER: 100% synthetic data.** Every capital market assumption,
> manager return stream, fund cash flow, and simulation in this repository is
> fabricated for demonstration purposes. Nothing here represents a real
> client, endowment, manager, fund, or investment recommendation. Do not use
> any output of this toolkit to make an actual investment decision.

A small Python toolkit that models the core analytical workflow of an
**OCIO (outsourced Chief Investment Officer)** serving nonprofit
endowments and foundations: setting long-term capital market assumptions,
building a policy portfolio, projecting spending capacity, evaluating
managers, keeping the portfolio on target, and reporting results back to a
client's investment committee. It was built to demonstrate the kind of
quantitative and communication skills relevant to a Client CIO Analyst role
supporting nonprofit endowment/foundation clients.

## Why these six modules?

An OCIO's job, boiled down, is a loop that runs every quarter for every
client:

1. **Set assumptions** about how asset classes will behave over the next
   decade (`capital_market_assumptions.py`).
2. **Translate those assumptions into a policy portfolio** — the strategic,
   long-term asset allocation the client's committee approves
   (`strategic_asset_allocation.py`).
3. **Check that the policy can support the client's spending** — most
   endowments and foundations exist to fund an annual budget, so the
   portfolio has to balance growth against a predictable payout
   (`spending_policy_model.py`).
4. **Hire and monitor the managers** who implement the policy portfolio,
   both liquid (mutual-fund-like) and illiquid (private fund) strategies
   (`manager_evaluation.py`).
5. **Keep the actual portfolio close to the policy portfolio** as markets
   move and some holdings can't be traded on demand
   (`portfolio_rebalancing.py`).
6. **Explain all of the above to the client** in a quarterly report that
   ties positioning, performance, and spending together
   (`client_reporting.py`).

Each module is runnable and testable on its own; `run_demo.py` chains all six
together into one end-to-end walkthrough.

## Concepts, in plain language

If you don't work in institutional investing day to day, here's what the
jargon in this repo means.

**Policy portfolio.** The long-term target mix of asset classes (e.g. "60%
public equity, 25% fixed income, 15% private equity") that a nonprofit's
investment committee formally adopts. It's a strategic decision, revisited
every few years — not something that changes because markets had a bad
month. Everything else in an OCIO relationship (manager selection,
rebalancing, performance reporting) is measured *against* the policy
portfolio.

**Capital market assumptions (CMA).** Forward-looking estimates of what each
asset class is expected to return, how volatile it will be, and how it
co-moves with every other asset class over the next ~10 years. CMAs are the
raw ingredients for building a policy portfolio: change the assumptions and
the "optimal" portfolio changes with them.

**Mean-variance optimization / efficient frontier.** For a given set of CMAs,
there's a curve of portfolios that give the *most* expected return for a
given level of risk (or equivalently, the *least* risk for a given return).
That curve is the efficient frontier. A real OCIO doesn't sit exactly on the
mathematically optimal frontier — it adds constraints (no shorting, caps on
illiquid holdings, a minimum cash buffer) that reflect what a nonprofit board
will actually approve and what the portfolio can operationally support.

**Black-Litterman.** A technique for blending two things: (1) the "market
prior" — the return assumptions implied by working backward from a market-
like starting portfolio, and (2) the OCIO's own "house views" — e.g. "we
think emerging markets will outperform investment-grade bonds by more than
the market implies." The output is a blended set of expected returns that
tilts the market prior toward the house view, with the size of the tilt
controlled by how confident the view is. This avoids the common problem
where an unconstrained optimizer takes a single confident opinion and turns
it into an all-or-nothing bet.

**Spending policy.** Most endowments/foundations exist to fund an ongoing
budget (scholarships, grants, operations), so the portfolio isn't just
"grow the money" — it has to support a sustainable, predictable annual draw.
A common approach is a **smoothed spending rule**: each quarter, spend a
fixed percentage (e.g. 4.5% annualized) of the *trailing 12-quarter (3-year)
average* market value, rather than of the current value. Averaging over
three years means a bad quarter in the market doesn't immediately force a
budget cut, and a great quarter doesn't immediately inflate spending — it
smooths the institution's cash flow at the cost of some lag.

**Manager due diligence — liquid strategies.** For managers investing in
public markets, OCIOs track risk-adjusted performance versus a benchmark:
*tracking error* (how much the manager's returns deviate from the
benchmark), *information ratio* (active return earned per unit of tracking
error — skill, not just risk-taking), and *up/down capture* (does the
manager keep more of the benchmark's gains than its losses?).

**Manager due diligence — private funds (PE/VC).** Private funds don't have
a daily NAV; they draw down committed capital over years and return it
(hopefully with a profit) even later, in an irregular pattern often called a
**J-curve** (early cash outflows, a dip in reported value, then later
inflows). The standard scorecard is:
- **IRR** — the annualized rate of return that makes the cash flows'
  present value zero, accounting for exact timing.
- **TVPI** (Total Value to Paid-In) — (money distributed back + current
  fund value) ÷ money invested. "For every dollar I put in, what do I have
  now (realized + unrealized)?"
- **DPI** (Distributions to Paid-In) — money actually distributed back ÷
  money invested. The realized-only slice of TVPI — cash in hand.
- **RVPI** (Residual Value to Paid-In) — remaining (unrealized) fund value ÷
  money invested. TVPI = DPI + RVPI.
- **PME** (Public Market Equivalent) — because IRR/TVPI alone don't say
  whether the fund actually beat a public-market alternative, PME reinvests
  every fund cash flow into a public index instead and compares final
  values. This toolkit implements the **Kaplan-Schoar PME (KS-PME)**: grow
  every contribution and distribution to a common valuation date at the
  public index's actual return, then compare the two totals. A KS-PME above
  1.0 means the private fund beat putting the same dollars, on the same
  schedule, into the public market.

**Rebalancing.** Markets move at different rates, so an actual portfolio
drifts away from its policy targets over time. A **rebalancing band** policy
resets a holding back toward its target once it drifts beyond a threshold
(e.g. ±3 percentage points). The catch for an endowment: private
equity/venture and hedge fund allocations usually **can't** be traded on
demand — you can't sell down an overweight PE stake to fund a rebalance the
way you can with public equities. This toolkit models that explicitly: only
the liquid sleeve of the portfolio is rebalanced, and illiquid weights are
left to drift (optionally nudged over time by directing new contributions
toward whatever's most underweight, since fresh cash doesn't have the same
liquidity constraint as an existing holding).

**Brinson attribution.** At the end of a quarter, a committee wants to know
*why* the portfolio beat or missed its benchmark. The classic
**Brinson-Hood-Beebower (BHB)** model splits total active return into three
pieces that add up exactly to the total:
- **Allocation effect** — value added (or lost) by weighting an asset class
  differently than the benchmark (e.g. being overweight equities when
  equities rallied).
- **Selection effect** — value added (or lost) by manager/security choice
  *within* an asset class (e.g. the equity manager beat the equity
  benchmark).
- **Interaction effect** — the cross-term between the two (being overweight
  an asset class *and* outperforming within it compounds; being overweight
  and underperforming compounds against you).

  `allocation + selection + interaction = total active return`, exactly —
  this reconciliation is directly unit-tested in this repo.

## Module map

| Module | What it does |
|---|---|
| `capital_market_assumptions.py` | Synthetic 10-year expected return/volatility/correlation assumptions across a 9-asset-class endowment universe (US/non-US developed/EM equity, core fixed income, private equity/venture, hedge funds diversifying & directional, real assets/real estate, cash). The correlation matrix is built from a latent-factor model so it's guaranteed positive semi-definite by construction. |
| `strategic_asset_allocation.py` | Mean-variance optimizer (SciPy SLSQP) with no-shorting, an illiquidity cap on PE + hedge funds, a private-equity-only cap, and a minimum cash buffer. Builds an efficient frontier and three named policy portfolios (Conservative/Moderate/Growth), each with expected return, volatility, Sharpe ratio, and a Monte-Carlo-estimated expected max drawdown. Includes an optional Black-Litterman layer for blending house views into the market-implied prior. |
| `spending_policy_model.py` | Smoothed (trailing 12-quarter average) spending rule; 20-year Monte Carlo projection of real (inflation-adjusted) endowment value and spending, reporting probability of real value preservation and spending volatility. |
| `manager_evaluation.py` | Liquid-strategy metrics (rolling Sharpe, tracking error, information ratio, up/down capture) and illiquid-strategy metrics (IRR, TVPI, DPI, RVPI, KS-PME) from synthetic manager returns and PE/VC J-curve cash flows. |
| `portfolio_rebalancing.py` | Simulates weight drift under differential asset-class returns, with illiquid holdings excluded from direct rebalancing trades. Compares a rebalancing-band policy (plus optional cash-flow-based rebalancing) against a no-rebalancing counterfactual via tracking error to the target. |
| `client_reporting.py` | Assembles a quarterly-report-style bundle: positioning vs. targets, Brinson-Hood-Beebower performance attribution, spending policy status, and an auto-generated plain-language executive summary. |

## Quickstart

```bash
pip install -r requirements.txt

# Run the full test suite
pytest tests/ -v

# Run the end-to-end demo (chains all six modules together)
python run_demo.py
```

## Testing approach

The test suite (`tests/`, ~54 tests) is built around correctness checks that
matter for this domain, not just code coverage:

- The CMA correlation matrix is symmetric, has a unit diagonal, and is
  **positive semi-definite** (a real requirement — an invalid correlation
  matrix produces nonsensical portfolio variance).
- Every efficient-frontier and policy-portfolio result is checked against
  its constraints (weights sum to 1, no shorting, illiquidity caps, minimum
  liquidity buffer).
- The spending-policy smoothing formula is checked against hand-computed
  trailing averages.
- IRR, TVPI/DPI/RVPI, and KS-PME are checked against hand-calculated cash
  flow examples with known closed-form answers.
- Brinson attribution is checked to **reconcile exactly**
  (allocation + selection + interaction = total active return), including
  over 20 rounds of randomized inputs.

## Design notes

- All optimization uses `scipy.optimize.minimize` (SLSQP) rather than a
  third-party convex solver, to keep the dependency footprint to
  numpy/pandas/scipy/pytest.
- Every module carries an explicit `DISCLAIMER` constant and every synthetic
  data generator is seeded for reproducibility.
- Type hints and docstrings are used throughout; `mypy` and `ruff` are both
  clean against the `endowment_ocio_toolkit` package.

## Reproducibility

### Data availability

No external data is downloaded. Inputs are generated or parameterised inside the code.

### Computational requirements

- Python 3.12.3. Every dependency version is pinned in `uv.lock`.
- Checked on Linux x86_64 with 2 CPU cores and 8 GB RAM. No GPU is needed.

```bash
pip install uv     # once
uv sync            # installs the exact pinned environment
uv run pytest -q   # runs the test suite
```

### Runtime

The test suite (54 tests) finishes in about 15 seconds on the machine above, including any data downloads it triggers.

### Citation

Citation metadata is in `CITATION.cff`. GitHub turns it into a Cite this repository button on the repo page.

