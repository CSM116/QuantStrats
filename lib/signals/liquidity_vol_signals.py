"""
liquidity_volume_signals.py

Generates volume-based and liquidity-driven signals for equity trading.
Includes tools for detecting abnormal volume activity, price-volume divergence,
and accumulation/distribution patterns.
"""

import pandas as pd
import numpy as np

def volume_spike(volume_series, window=20, threshold=2.0):
    """
    Flags volume spikes: volume > threshold × rolling mean.
    Indicates increased interest or potential breakout.
    """
    rolling_avg = volume_series.rolling(window).mean()
    return volume_series > threshold * rolling_avg

def volume_price_divergence(price_series, volume_series, window=20):
    """
    Measures divergence between price change and volume change direction.
    Positive when price up but volume down (and vice versa).
    Can indicate false moves or exhaustion.
    """
    price_ret = price_series.pct_change()
    volume_ret = volume_series.pct_change()
    return price_ret.rolling(window).corr(volume_ret) * -1

def accumulation_distribution(close, high, low, volume):
    """
    Computes the Accumulation/Distribution Line (ADL).
    Useful to detect underlying buying/selling pressure.
    """
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    ad = (clv * volume).fillna(0).cumsum()
    return ad
