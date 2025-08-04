
"""
risk_signals.py

Contains functions for generating volatility and risk-based signals, including
realized volatility ratios, implied-realized spreads, rolling beta, and idiosyncratic volatility.
Useful for regime detection, signal filtering, and risk-aware forecasting.
"""


import pandas as pd
import numpy as np


def vol_contraction_expansion(price, short_window=20, long_window=60):
    """
    Measures realized vol shift.
    >1 = volatility expansion, <1 = contraction.
    Used to time breakouts or fade mean reversion.
    """
    short_vol = price.pct_change().rolling(short_window).std()
    long_vol = price.pct_change().rolling(long_window).std()
    return short_vol / long_vol



def implied_realized_spread(implied_vol, price, window=20):
    """
    Spread between implied and realized vol over `window`.
    Positive = market overpricing risk; potential vol crush.
    Negative = underpricing risk; potential breakout.
    """
    realized_vol = price.pct_change().rolling(window).std()
    return implied_vol - realized_vol



def vol_zscore(price, window=20, lookback=60):
    """
    Z-score of short-term realized vol vs historical vol.
    Helps identify abnormal compression or spikes.
    """
    short_vol = price.pct_change().rolling(window).std()
    return (short_vol - short_vol.rolling(lookback).mean()) / short_vol.rolling(lookback).std()


def rolling_beta(stock_returns, market_returns, window=60):
    """
    Measures sensitivity of stock to market.
    A rising beta can indicate increased systemic risk or breakout exposure.
    """
    cov = stock_returns.rolling(window).cov(market_returns)
    var = market_returns.rolling(window).var()
    return cov / var


def idiosyncratic_volatility(stock_returns, market_returns, window=60):
    """
    Residual volatility after removing market beta.
    High idiosyncratic vol = potential for re-pricing due to stock-specific news.
    """
    beta = rolling_beta(stock_returns, market_returns, window)
    fitted = beta * market_returns
    residual = stock_returns - fitted
    return residual.rolling(window).std()


def vol_of_vol(series, vol_window=20):
    """
    Computes the volatility of the rolling volatility (realized vol).
    Used to detect changes in risk regime or signal instability.
    """
    vol = series.pct_change().rolling(vol_window).std()
    return vol.rolling(vol_window).std()


def vol_breakout(series, window=60, threshold=0.75):
    """
    Flags when volatility exceeds a high percentile (e.g., 75th).
    Can signal breakout opportunities.
    """
    vol = series.pct_change().rolling(window).std()
    return vol > vol.rolling(window).quantile(threshold)






