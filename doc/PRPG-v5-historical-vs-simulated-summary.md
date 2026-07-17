# PRPG v5 historical versus simulated return summary

**Status:** descriptive report only; this is not a release gate or a hypothesis test.

The simulated columns use every delivered monthly observation from all 5,000 fifty-year paths. Historical values use the 219 complete synchronized months from April 2008 through June 2026.

## Core return and volatility comparison

| Role (ticker) | Historical geometric return | Simulated pooled geometric return | Historical volatility | Simulated volatility |
|---|---:|---:|---:|---:|
| Equity (ACWI) | 8.59% | 8.05% | 16.81% | 16.99% |
| Muni Bond (MUB) | 3.20% | 3.24% | 5.06% | 5.18% |
| Taxable Bond (LQD) | 4.13% | 4.11% | 8.35% | 8.47% |

## Distribution of 50-year path annualized returns

| Role | 5th percentile | Median | 95th percentile |
|---|---:|---:|---:|
| Equity | 3.39% | 8.17% | 12.55% |
| Muni Bond | 1.99% | 3.24% | 4.48% |
| Taxable Bond | 2.03% | 4.14% | 6.18% |

## Monthly tail comparison

| Role | History 1% | Simulated 1% | History 5% | Simulated 5% | History 95% | Simulated 95% | History 99% | Simulated 99% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Equity | -12.10% | -12.26% | -8.07% | -8.11% | 8.39% | 8.89% | 11.17% | 11.32% |
| Muni Bond | -3.04% | -3.05% | -2.52% | -2.55% | 2.47% | 2.50% | 4.65% | 4.84% |
| Taxable Bond | -6.25% | -6.31% | -3.25% | -3.27% | 3.75% | 3.77% | 6.38% | 6.65% |

## Correlation comparison

These are pooled monthly log-return correlations; the canonical hard validator separately reports full-population daily correlations.

| Pair | Historical | Simulated pooled |
|---|---:|---:|
| equity / muni bond | 0.331 | 0.335 |
| equity / taxable bond | 0.519 | 0.518 |
| muni bond / taxable bond | 0.728 | 0.733 |

## Drawdown comparison

| Role | Historical 219m | Simulated first 219m median (5%-95%) | Simulated full 600m median (5%-95%) |
|---|---:|---:|---:|
| Equity | 51.70% | 41.78% (23.95%-67.01%) | 54.02% (36.10%-76.10%) |
| Muni Bond | 11.60% | 10.94% (6.73%-19.42%) | 14.45% (9.69%-23.29%) |
| Taxable Bond | 23.26% | 19.72% (11.19%-35.28%) | 26.14% (16.86%-41.73%) |

## Interpretation

- The close return, volatility, and correlation values show what the synchronized empirical-month sampler preserves well at the ensemble level.
- Tail values remain tied to observed source months; the HMM changes their ordering and persistence rather than inventing unseen shocks.
- Matched-horizon drawdowns are the fairer historical comparison. Fifty-year drawdowns are naturally deeper because the simulated horizon is much longer.
- The comparison is conditional on one 219-month ETF history and one fitted K4 model. It does not measure parameter uncertainty or prove future realism.
- Cross-month dependence is approximated by the K4 transition chain. Within each sampled month, exact synchronized daily returns are retained.
