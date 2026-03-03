import numpy as np
from pricing import black_scholes_price
from greeks import delta as bs_delta


def simulate_delta_hedge(S0, K, T, r, sigma_model, sigma_real, option_type="call",
                          hedge_frequency=21, n_simulations=200, n_steps=252, seed=42):
    """
    Simulate dynamic delta hedging strategy.

    sigma_model: vol used to compute hedge ratios (what the trader assumes)
    sigma_real:  vol the market actually realizes (path generation)
    hedge_frequency: how often to rebalance (in trading days, e.g. 1=daily, 5=weekly, 21=monthly)

    Returns dict with P&L stats and sample paths for visualization.
    """
    np.random.seed(seed)
    dt = T / n_steps
    rebalance_days = set(range(0, n_steps, hedge_frequency))

    all_pnl = []
    sample_paths = []  # store a subset for visualization

    # Initial option premium received (short call/put)
    option_premium = black_scholes_price(S0, K, T, r, sigma_model, option_type)

    for sim in range(n_simulations):
        # Generate spot path using REAL vol
        Z = np.random.standard_normal(n_steps)
        log_rets = (r - 0.5 * sigma_real ** 2) * dt + sigma_real * np.sqrt(dt) * Z
        S_path = S0 * np.exp(np.cumsum(np.hstack([0, log_rets])))

        cash = option_premium   # start with premium received
        hedge_position = 0.0   # units of stock held
        total_stock_cost = 0.0

        path_data = {"spot": S_path, "delta": [], "pnl_running": []}

        for step in range(n_steps):
            S = S_path[step]
            t_remaining = T - step * dt

            if step in rebalance_days:
                # Compute model delta at current spot/time
                new_delta = bs_delta(S, K, max(t_remaining, 1e-6), r, sigma_model, option_type)
                delta_change = new_delta - hedge_position

                # Buy/sell stock to rebalance — cost comes from cash account
                trade_cost = delta_change * S
                cash -= trade_cost
                total_stock_cost += abs(trade_cost)
                hedge_position = new_delta

                # Accrue interest on cash
                cash *= np.exp(r * dt)

            path_data["delta"].append(hedge_position)
            path_data["pnl_running"].append(cash + hedge_position * S_path[step])

        # At expiry: close stock position, settle option payoff
        S_T = S_path[-1]
        cash += hedge_position * S_T  # sell stock

        if option_type == "call":
            option_payoff = max(S_T - K, 0)
        else:
            option_payoff = max(K - S_T, 0)

        cash -= option_payoff  # pay out option
        final_pnl = cash
        all_pnl.append(final_pnl)

        if sim < 50:  # store first 50 paths for viz
            path_data["final_pnl"] = final_pnl
            sample_paths.append(path_data)

    all_pnl = np.array(all_pnl)
    return {
        "pnl_mean": all_pnl.mean(),
        "pnl_std": all_pnl.std(),
        "pnl_all": all_pnl,
        "sample_paths": sample_paths,
        "option_premium": option_premium,
        "sharpe": all_pnl.mean() / (all_pnl.std() + 1e-9),
    }
