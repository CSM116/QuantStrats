
"""

Provides risk metrics for strategy evaluation, including:
- Cumulative returns
- Sharpe and Sortino ratios
- Max drawdown
- Rolling volatility

Inputs are typically daily returns from backtest outputs.
"""


import pandas as pd
import numpy as np


def cumulative_returns(returns):
    """
    Compound returns over time.
    Useful for plotting and comparing growth.
    """
    return (1 + returns).cumprod() - 1



def sharpe_ratio(returns, freq=252):
    """
    Return/risk metric. Higher is better.
    Annualized assuming freq = trading days.
    """
    excess_ret = returns.mean() * freq
    risk = returns.std() * np.sqrt(freq)
    return excess_ret / risk if risk != 0 else np.nan



def max_drawdown(cum_returns):
    """
    Maximum % drop from a prior peak.
    Measures downside risk.
    """
    peak = cum_returns.cummax()
    drawdown = cum_returns - peak
    return drawdown.min()



def annualized_return(returns, freq=252):
    """
    Compound annual growth rate (CAGR).
    """
    return (1 + returns.mean()) ** freq - 1



def win_rate(returns):
    """
    Percent of days with positive return.
    Indicates consistency.
    """
    return (returns > 0).mean()



def rolling_sharpe(returns, window=60, freq=252):
    """
    Measures consistency of Sharpe through time.
    """
    roll_mean = returns.rolling(window).mean() * freq
    roll_std = returns.rolling(window).std() * np.sqrt(freq)
    return roll_mean / roll_std


#################### ✅ Risk-Adjusted Metrics (analytics.py additions) ####################

def sortino_ratio(returns, freq=252):
    """
    Like Sharpe but penalizes only downside volatility.
    Higher = better risk-adjusted return.
    """
    downside_std = returns[returns < 0].std() * np.sqrt(freq)
    expected_ret = returns.mean() * freq
    return expected_ret / downside_std if downside_std != 0 else np.nan


def calmar_ratio(returns):
    """
    Annualized return / absolute max drawdown.
    Useful for comparing risk-return efficiency.
    """
    ann_ret = annualized_return(returns)
    max_dd = abs(max_drawdown((1 + returns).cumprod() - 1))
    return ann_ret / max_dd if max_dd != 0 else np.nan


def rolling_volatility(returns, window=60, freq=252):
    """
    Measures how volatile returns are over time.
    """
    return returns.rolling(window).std() * np.sqrt(freq)



