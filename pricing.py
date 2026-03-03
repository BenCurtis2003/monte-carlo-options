import numpy as np
from scipy.stats import norm


def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes analytical option price.
    S: spot price, K: strike, T: time to expiry (years),
    r: risk-free rate, sigma: volatility, option_type: 'call' or 'put'
    """
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


def monte_carlo_price(S, K, T, r, sigma, option_type="call", n_simulations=10000, n_steps=252, seed=42):
    """
    Monte Carlo option pricing using GBM paths.
    Returns price estimate, std error, and all final spot prices.
    """
    np.random.seed(seed)
    dt = T / n_steps
    
    # Simulate GBM paths — shape (n_simulations, n_steps+1)
    Z = np.random.standard_normal((n_simulations, n_steps))
    log_returns = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    
    # Build full price paths
    log_paths = np.cumsum(log_returns, axis=1)
    log_paths = np.hstack([np.zeros((n_simulations, 1)), log_paths])
    paths = S * np.exp(log_paths)

    # Payoffs at expiry
    S_T = paths[:, -1]
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)

    discounted = np.exp(-r * T) * payoffs
    price = discounted.mean()
    stderr = discounted.std() / np.sqrt(n_simulations)

    return price, stderr, paths


def implied_volatility(market_price, S, K, T, r, option_type="call", tol=1e-6, max_iter=200):
    """
    Newton-Raphson implied volatility solver.
    Returns IV or None if it fails to converge.
    """
    from greeks import vega

    sigma = 0.2  # initial guess
    for _ in range(max_iter):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        v = vega(S, K, T, r, sigma)
        if abs(v) < 1e-10:
            return None
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / v
        if sigma <= 0:
            return None
    return None
