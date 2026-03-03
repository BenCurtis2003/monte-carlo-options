# Monte Carlo Option Pricing Engine

An interactive options analytics dashboard built with Python and Plotly Dash, covering Monte Carlo simulation, Black-Scholes pricing, Greeks analysis, and dynamic delta hedging.

---

## Features

### Pricing
- **Monte Carlo simulation** — GBM-based path simulation with configurable number of runs
- **Black-Scholes analytical pricing** — closed-form benchmark
- **Implied volatility solver** — Newton-Raphson IV extraction from market prices
- **Vol edge display** — real-time spread between model vol and market-implied vol (the core of flow trading)

### Greeks Dashboard
- Delta, Gamma, Vega, Theta computed analytically across the full vol surface
- Interactive slider to observe Greeks sensitivity to model assumptions

### Dynamic Delta Hedging Simulator
- Simulates a short option position delta-hedged at configurable rebalance frequencies (daily → monthly)
- Separates **model vol** (hedge ratio) from **realized vol** (path) to show gamma/theta P&L bleed
- P&L distribution with mean, std, and Sharpe across 300 simulated paths

### Implied Vol Surface
- Simulated vol surface with realistic skew and term structure
- 3D interactive surface via Plotly

---

## Getting Started

```bash
git clone https://github.com/yourusername/monte-carlo-options.git
cd monte-carlo-options
pip install -r requirements.txt
python app.py
```

Open `http://localhost:8050` in your browser.

---

## Project Structure

```
├── app.py          # Dash app layout and callbacks
├── pricing.py      # Black-Scholes + Monte Carlo engine + IV solver
├── greeks.py       # Delta, Gamma, Vega, Theta, Rho
├── hedging.py      # Dynamic delta hedging P&L simulator
└── requirements.txt
```

---

## Key Concepts

| Concept | Implementation |
|---|---|
| GBM path simulation | `pricing.py → monte_carlo_price()` |
| Greeks (analytical) | `greeks.py` |
| IV via Newton-Raphson | `pricing.py → implied_volatility()` |
| Delta hedge rebalancing | `hedging.py → simulate_delta_hedge()` |
| Vol edge (mkt vs model) | `app.py → update() → edge` |

---

## Why This Matters for Flow Trading

Flow desks price and risk-manage large option books in real time. The three core questions are:

1. **What is this worth?** → BS vs MC pricing comparison
2. **What is my risk?** → Greeks dashboard
3. **How well can I hedge it?** → Delta hedging P&L simulator

The vol edge metric (`σ_market − σ_model`) represents the core alpha opportunity in options market making — buying cheap vol or selling rich vol relative to expected realized volatility.

---

## Tech Stack

- `Python 3.11`
- `Plotly Dash` — interactive UI
- `NumPy / SciPy` — simulation and optimization
