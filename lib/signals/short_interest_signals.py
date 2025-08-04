
"""
Short Interest & Crowding Module

Extracts short positioning signals to detect crowding and squeeze risk.
Includes short interest ratio, days to cover, z-score, change, and squeeze index.

Interpretation:
- High values = bearish crowding or squeeze potential
- Low values = low short pressure or complacency

Useful for alpha signals in 20–60 day strategies.
"""


import pandas as pd
import numpy as np



def short_interest_ratio(short_interest, float_shares):
    """
    Short Interest Ratio: short interest / float.
    High values suggest bearish sentiment or squeeze potential.
    """
    si = _validate_series(short_interest, "short_interest")
    fl = _validate_series(float_shares, "float_shares")
    return si / fl.replace(0, np.nan)

def days_to_cover(short_interest, avg_daily_volume):
    """
    Days to Cover: short interest / average daily volume.
    Reflects time to unwind short positions.
    """
    si = _validate_series(short_interest, "short_interest")
    vol = _validate_series(avg_daily_volume, "avg_daily_volume")
    return si / vol.replace(0, np.nan)

def short_interest_change(short_interest, window=2):
    """
    Short Interest Change: SI - SI.shift(window).
    Captures crowding dynamics over recent periods.
    """
    si = _validate_series(short_interest, "short_interest")
    return si - si.shift(window)

def short_interest_zscore(short_interest, window=60):
    """
    Short Interest Z-Score: (SI - mean) / std over rolling window.
    Detects unusual crowding.
    """
    si = _validate_series(short_interest, "short_interest")
    rolling_mean = si.rolling(window).mean()
    rolling_std = si.rolling(window).std()
    return (si - rolling_mean) / rolling_std.replace(0, np.nan)

def squeeze_risk_index(short_interest, avg_daily_volume, price, lookback=10):
    """
    Squeeze Risk Index: estimates short squeeze potential.
    Combines:
    - Days to Cover (position buildup)
    - Illiquidity (inverse volume)
    - Price strength (positive returns)
    """
    si = _validate_series(short_interest, "short_interest")
    vol = _validate_series(avg_daily_volume, "avg_daily_volume")
    px = _validate_series(price, "price")

    dtc = si / vol.replace(0, np.nan)
    illiquidity = 1 / vol.replace(0, np.nan)
    strength = px.pct_change(lookback).clip(lower=0)
    
    return dtc * illiquidity * strength

