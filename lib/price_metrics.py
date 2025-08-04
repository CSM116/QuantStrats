"""
Contains stateless functions for calculating raw price-based technical indicators,
such as RSI, MACD, Bollinger Bands, Z-score, and smoothed momentum.

Each function takes a pandas Series of prices and returns a Series or tuple of Series.
"""

import pandas as pd
import numpy as np









def rsi(series, window=14):
    """
    Relative Strength Index: momentum oscillator between 0–100.
    RSI > 70 indicates overbought; RSI < 30 indicates oversold.
    Used to identify potential trend reversals.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def bollinger_bands(series, window=20, num_std=2):
    """
    Bollinger Bands: volatility bands around a moving average.
    - Upper band = MA + k*std
    - Lower band = MA - k*std
    Price touching outer bands may indicate overbought/oversold conditions.
    """
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def smoothed_momentum(series, span=20):
    """
    Smoothed momentum: difference between current and previous price,
    exponentially weighted to highlight trend direction.
    Positive values = upward momentum; negative = downward momentum.
    """
    momentum = series - series.shift(1)
    return momentum.ewm(span=span, adjust=False).mean()


def macd(series, fast=12, slow=26, signal=9):
    """
    MACD: difference between fast and slow EMAs of price.
    - MACD line = EMA(fast) - EMA(slow)
    - Signal line = EMA(MACD, signal)
    Positive MACD = bullish momentum; negative = bearish.
    Crossovers with signal line are used as trade signals.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def distance_from_ma(series, window=20):
    """
    Measures the % distance of price from its moving average.
    Useful for reversion setups.
    """
    ma = series.rolling(window).mean()
    return (series - ma) / ma
    
