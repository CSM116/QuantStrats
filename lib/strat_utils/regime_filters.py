"""
regime_filters.py

Implements filters and indicators to detect market regimes (e.g., trending, mean-reverting, volatile).
Used to condition or adjust signals based on macro or statistical environments.
"""

import pandas as pd
import numpy as np

def volatility_regime(price_series, fast=20, slow=60):
    """
    Compares fast vs. slow realized volatility.
    Returns 1 if fast vol > slow vol (high vol regime), else 0.
    """
    fast_vol = price_series.pct_change().rolling(fast).std()
    slow_vol = price_series.pct_change().rolling(slow).std()
    return (fast_vol > slow_vol).astype(int)


def trend_strength(price_series, lookback=60):
    """
    Computes R-squared from linear regression of price over time.
    Higher values indicate stronger trends.
    """
    def rsq(y):
        x = np.arange(len(y))
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        y_fit = m * x + c
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 0

    return price_series.rolling(lookback).apply(rsq, raw=True)


def macro_conditioning(signal_series, macro_series, threshold):
    """
    Applies macro filter: only pass signal if macro condition is met.
    Example: signal active only if interest rates < threshold.
    """
    return signal_series.where(macro_series < threshold, 0)
