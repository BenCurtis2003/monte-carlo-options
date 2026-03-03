import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def delta(S, K, T, r, sigma, option_type="call"):
    """Rate of change of option price with respect to spot."""
    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def gamma(S, K, T, r, sigma):
    """Rate of change of delta with respect to spot (same for call/put)."""
    if T <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """Sensitivity to volatility (per 1% move in vol)."""
    if T <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) * 0.01  # scaled to 1% vol move


def theta(S, K, T, r, sigma, option_type="call"):
    """Time decay — value lost per calendar day."""
    if T <= 0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        return (term1 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        return (term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365


def rho(S, K, T, r, sigma, option_type="call"):
    """Sensitivity to risk-free rate."""
    if T <= 0:
        return 0.0
    _, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01


def greeks_surface(S, K, T, r, sigma_range, option_type="call"):
    """
    Compute Greeks across a range of vol assumptions.
    Returns dict of arrays for plotting.
    """
    deltas, gammas, vegas, thetas = [], [], [], []
    for s in sigma_range:
        deltas.append(delta(S, K, T, r, s, option_type))
        gammas.append(gamma(S, K, T, r, s))
        vegas.append(vega(S, K, T, r, s))
        thetas.append(theta(S, K, T, r, s, option_type))
    return {
        "sigma": sigma_range,
        "delta": deltas,
        "gamma": gammas,
        "vega": vegas,
        "theta": thetas,
    }
