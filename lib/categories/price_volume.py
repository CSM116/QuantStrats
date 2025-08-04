import pandas as pd
import numpy as np
from signals_utils import validate_series


def momentum_1m(price, window=21):
    """
    1-Month Momentum: percentage change over ~21 trading days.
    Positive momentum may indicate trend continuation.
    """
    p = validate_series(price, "price")
    return p.pct_change(periods=window)


def volume_spike(volume, window=20):
    """
    Volume Spike: volume / rolling average volume.
    Values >> 1 indicate abnormal trading interest.
    """
    v = validate_series(volume, "volume")
    avg_vol = v.rolling(window).mean()
    return v / avg_vol.replace(0, np.nan)


def price_gap(open_price, prev_close):
    """
    Price Gap: (today's open - previous close) / previous close.
    Captures overnight news or sentiment shifts.
    """
    o = validate_series(open_price, "open_price")
    c = validate_series(prev_close, "prev_close")
    return (o - c) / c.replace(0, np.nan)


def intraday_reversal(open_price, close_price):
    """
    Intraday Reversal: (close - open) / open.
    Positive = fade of negative open; Negative = fade of gap up.
    """
    o = validate_series(open_price, "open_price")
    cl = validate_series(close_price, "close_price")
    return (cl - o) / o.replace(0, np.nan)


def price_volume_trend(price, volume):
    """
    Price Volume Trend (PVT): cumulative volume * % price change.
    Measures volume-weighted momentum.
    """
    p = validate_series(price, "price")
    v = validate_series(volume, "volume")
    pct_change = p.pct_change().fillna(0)
    return (v * pct_change).cumsum()


def volatility_contraction(price, window=20):
    """
    Volatility Contraction: rolling std of returns.
    Lower values may precede breakouts.
    """
    p = validate_series(price, "price")
    returns = p.pct_change()
    return returns.rolling(window).std()


def price_breakout(price, window=20):
    """
    Price Breakout: price > rolling high.
    Flags breakout above recent range.
    """
    p = validate_series(price, "price")
    return p > p.rolling(window).max().shift(1)


def volume_breakout(volume, window=20):
    """
    Volume Breakout: volume > rolling max.
    Identifies surge in trading activity.
    """
    v = validate_series(volume, "volume")
    return v > v.rolling(window).max().shift(1)


def relative_volume(volume, dayofweek=None):
    """
    Relative Volume: today's volume vs. avg volume for that weekday.
    Helps detect abnormal flow controlling for weekday effects.
    """
    v = validate_series(volume, "volume")
    if dayofweek is None:
        dayofweek = v.index.dayofweek
    return v / v.groupby(dayofweek).transform('mean')

