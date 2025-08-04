
"""
signal_rules.py

Defines long/short signal generation rules using technical indicators from price_signals.
Each function returns a Series of +1, -1, or 0 representing long, short, or neutral positions.

Functions are designed to be composable and used in systematic strategy pipelines.
"""


import pandas as pd
from ..signals.price_signals import * 

def generate_zscore_signals(price, window=30, entry=1.5, exit=0.5):
    """
    Buy when Z-score < -entry, sell when Z-score > entry.
    Exit when Z-score crosses back toward 0.
    Works best in mean-reverting markets.
    """
    z = zscore(price, window)
    signal = pd.Series(0, index=price.index)
    signal[z < -entry] = 1
    signal[z > entry] = -1
    signal[(z > -exit) & (z < exit)] = 0
    return signal.ffill()


def generate_macd_signals(price, fast=12, slow=26, signal_window=9):
    """
    Long when MACD > Signal Line, short when MACD < Signal.
    Captures trend direction and changes.
    """
    macd_line, signal_line = macd(price, fast, slow, signal_window)
    signal = pd.Series(0, index=price.index)
    signal[macd_line > signal_line] = 1
    signal[macd_line < signal_line] = -1
    return signal

    
def generate_bollinger_reversal_signals(price, window=20, num_std=2):
    """
    Long when price closes below lower band, short when above upper.
    Mean-reversion strategy around volatility bands.
    """
    mid, upper, lower = bollinger_bands(price, window, num_std)
    signal = pd.Series(0, index=price.index)
    signal[price < lower] = 1
    signal[price > upper] = -1
    return signal

def generate_smoothed_momentum_signals(price, span=30, threshold=0):
    """
    Long if smoothed momentum > threshold, short if < -threshold.
    Captures persistent directional moves.
    """
    mom = smoothed_momentum(price, span)
    signal = pd.Series(0, index=price.index)
    signal[mom > threshold] = 1
    signal[mom < -threshold] = -1
    return signal


def filter_signals(base_signal, trend_signal):
    """
    Use trend signal to filter base signal.
    E.g., only take long mean-reversion signals in uptrend.
    """
    return base_signal.where(trend_signal == 1, 0)











