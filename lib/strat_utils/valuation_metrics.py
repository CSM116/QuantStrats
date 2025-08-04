
"""
Relative Valuation Module

Generates fundamental valuation ratios for systematic strategies.
Includes P/E, P/B, P/S, EV/EBIT, and valuation z-scores.

Interpretation:
- High values = potentially overvalued
- Low values = potentially undervalued

Useful for cross-sectional and historical mean-reversion signals.
"""

from signals_utils import *


import pandas as pd
import numpy as np



def price_to_earnings(price, earnings_per_share):
    """
    Price-to-Earnings (P/E): price / EPS.
    Lower values may indicate undervaluation relative to peers or history.
    """
    p = _validate_series(price, "price")
    eps = _validate_series(earnings_per_share, "earnings_per_share")
    return p / eps.replace(0, np.nan)


def price_to_book(price, book_value_per_share):
    """
    Price-to-Book (P/B): price / book value per share.
    Lower values suggest potential undervaluation; varies by sector.
    """
    p = _validate_series(price, "price")
    bvps = _validate_series(book_value_per_share, "book_value_per_share")
    return p / bvps.replace(0, np.nan)


def price_to_sales(price, revenue_per_share):
    """
    Price-to-Sales (P/S): price / revenue per share.
    Useful for growth stocks with low or negative earnings.
    """
    p = _validate_series(price, "price")
    rps = _validate_series(revenue_per_share, "revenue_per_share")
    return p / rps.replace(0, np.nan)


def ev_to_ebit(enterprise_value, ebit):
    """
    EV/EBIT: enterprise value / EBIT.
    Used to compare firms regardless of capital structure.
    """
    ev = _validate_series(enterprise_value, "enterprise_value")
    e = _validate_series(ebit, "ebit")
    return ev / e.replace(0, np.nan)


def valuation_zscore(valuation_series, window=60):
    """
    Valuation Z-Score: standardizes valuation relative to historical range.
    High = overvalued vs history; Low = undervalued.
    """
    v = _validate_series(valuation_series, "valuation_series")
    mean = v.rolling(window).mean()
    std = v.rolling(window).std()
    return (v - mean) / std.replace(0, np.nan)

