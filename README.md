# Monte Carlo Simulation Framework : Quantitative Finance
### *Portfolio Risk Estimation · VaR/CVaR · Scenario Analysis · Multi-Asset Correlated GBM*

---

## Overview

A production-grade Monte Carlo simulation engine built for quantitative trading and risk management. The project models portfolio dynamics using Geometric Brownian Motion (GBM), computes tail-risk metrics (VaR, CVaR), and stress-tests portfolios across multiple volatility regimes. 


---

## Problem Statement

**How much capital is at risk over a 1-year horizon, and how does tail risk escalate under stress?**

Classical closed-form risk models (e.g. parametric VaR) assume normality and constant volatility, which do not hold in real markets. Monte Carlo simulation relaxes these constraints by:

- Generating thousands of stochastic price paths under calibrated dynamics
- Capturing the full empirical return distribution (including fat tails and skew)
- Enabling stress testing across arbitrary volatility regimes
- Modelling realistic cross-asset correlation without closed-form tractability constraints

**Why GBM?**
GBM is the foundation of the Black-Scholes framework and widely used for equity and index modelling. It ensures non-negative prices (log-normal), incorporates both drift and diffusion, and has a clean closed-form solution enabling fast vectorised simulation.

---

## Project Structure: 

```
monte_carlo_quant/
│
├── monte_carlo_quant.py       # Core simulation engine (all modules)
├── mc_dashboard.png           # Single-asset 4-panel risk dashboard
├── mc_scenarios.png           # Volatility scenario overlays
├── mc_portfolio.png           # Multi-asset correlated portfolio output
└── README.md                  # This file
```

---

## Simulation Parameters

| Parameter         | Symbol | Value          | Description                        |
|-------------------|--------|----------------|------------------------------------|
| Initial Price     | S₀     | $500.00        | SPY proxy starting price           |
| Drift             | μ      | 8% p.a.        | Expected annualised return         |
| Volatility        | σ      | 18% p.a.       | Annualised price volatility        |
| Time Horizon      | T      | 1 year         | Simulation duration                |
| Time Step         | dt     | 1/252          | Daily rebalancing frequency        |
| Simulations       | N      | 10,000         | Monte Carlo path count             |
| Portfolio Value   | —      | $1,000,000     | AUM for dollar-denominated metrics |
| Random Seed       | —      | 42             | Full reproducibility               |

---

## Key Outputs

### Single-Asset Risk Report (SPY Proxy)
| Metric             | Value   |
|--------------------|---------|
| E[Return]          | ~8.0%   |
| Std(Return)        | ~18.5%  |
| VaR 99% (return)   | ~-35%   |
| CVaR 99% (return)  | ~-41%   |
| VaR 99% ($)        | ~-$350K |
| CVaR 99% ($)       | ~-$410K |
| Mean Max Drawdown  | ~-21%   |

### Scenario Analysis (VaR 99%)
| Scenario  | σ    | VaR 99%  | CVaR 99% |
|-----------|------|----------|----------|
| Base      | 18%  | ~-35%    | ~-41%    |
| Elevated  | 30%  | ~-52%    | ~-60%    |
| Crisis    | 55%  | ~-76%    | ~-82%    |
| Meltdown  | 80%  | ~-89%    | ~-93%    |

### Multi-Asset Portfolio (4-Asset Correlated)
| Asset        | μ    | σ    | Weight |
|--------------|------|------|--------|
| US Equity    | 8%   | 18%  | 40%    |
| EM Equity    | 10%  | 25%  | 20%    |
| Gov Bonds    | 3%   | 7%   | 30%    |
| Commodities  | 5%   | 22%  | 10%    |

Correlation: US/EM Equity ρ=0.75 · Equity/Bonds ρ=-0.30 (flight-to-safety)

---

## Technical Implementation

### Core Functions
```python
simulate_price_paths()          # Vectorised GBM via closed-form log-normal solution
calculate_returns()             # Terminal total-return computation
compute_var_cvar()              # Parametric VaR & Expected Shortfall (multi-level)
compute_max_drawdown()          # Vectorised rolling peak-to-trough drawdown
simulate_correlated_portfolio() # Cholesky decomposition for cross-asset correlation
scenario_analysis()             # Stress test sweep across volatility regimes
plot_simulations()              # 4-panel publication-quality risk dashboard
```

### Mathematical equation
- **GBM**:  `S(t) = S₀ · exp[(μ - σ²/2)·t + σ·√t·Z]`  where `Z ~ N(0,1)`
- **Cholesky**: `Z_corr = L · Z_indep` where `Σ = L·Lᵀ`
- **CVaR**: `CVaR(α) = E[Loss | Loss > VaR(α)]`

---

## Extensions

### Extension A : Correlated Multi-Asset Portfolio
Uses **Cholesky decomposition** of the correlation matrix to generate correlated Brownian motions, enabling realistic cross-asset dynamics (e.g. equity-bond flight-to-safety).

### Extension B : Scenario / Volatility Shock Analysis
Sweeps across four volatility regimes (Base → Elevated → Crisis → Meltdown) to quantify how tail risk escalates non-linearly ; critical for stress-testing regulatory capital requirements.

---

## Scalability Notes

| Approach              | Implementation                                       |
|-----------------------|------------------------------------------------------|
| **Vectorisation**     | NumPy broadcasting eliminates Python loops entirely  |
| **Parallel compute**  | `multiprocessing.Pool` or `joblib.Parallel` for N>1M |
| **GPU acceleration**  | CuPy drop-in for NumPy; PyTorch tensors for batching |
| **Cloud deployment**  | AWS Batch / GCP Cloud Run for on-demand execution    |
| **Distributed**       | Dask arrays; Ray for actor-based simulation trees    |

For 1M simulations: NumPy vectorised on CPU runs in ~8–12 s. GPU (CUDA) achieves ~50× speedup.

---

## Limitations

1. **Constant parameters**: GBM uses fixed μ and σ; real markets exhibit volatility clustering (GARCH effects), mean reversion, and regime shifts.
2. **Log-normal distribution**: Fat tails and negative skewness observed empirically are underrepresented; jump-diffusion models (Merton) address this.
3. **Static correlation**: Cholesky uses a fixed correlation matrix; correlations spike toward 1.0 in crises (correlation breakdown), which this model does not capture.
4. **Path independence**: GBM is Markovian; it does not model autocorrelation or momentum found in real asset returns.
5. **No transaction costs or market impact**: Relevant for execution-focused risk modelling.

---

## Installation & Usage

```bash
pip install numpy pandas matplotlib scipy

python monte_carlo_quant.py
```
