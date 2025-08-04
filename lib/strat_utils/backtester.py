
"""
backtester.py

Implements vectorized backtesting functions to simulate PnL from trading signals.
Includes functions for daily and multi-day holding period logic with optional transaction costs.

Assumes input signals are pre-aligned with the price series.
"""


import pandas as pd
import numpy as np


def backtest_signals(price, signal, hold=20, transaction_cost=0.0):
    """
    Executes a position for `hold` days after each signal change.
    Vectorized implementation; assumes signal is 1, 0, or -1.
    Returns daily strategy returns and held position.
    """
    signal = signal.shift(1)
    returns = price.pct_change().fillna(0)

    position = pd.Series(0, index=price.index)
    for i in range(len(signal)):
        if signal.iloc[i] != 0:
            position.iloc[i:i+hold] = signal.iloc[i]

    trades = signal.diff().abs().fillna(0)
    strategy_returns = position * returns - trades * transaction_cost
    return strategy_returns, position


def backtest_daily(price, signal, transaction_cost=0.0):
    """
    Simple backtest assuming daily position defined by signal.
    No holding logic. Best for comparing strategies.
    """
    returns = price.pct_change().fillna(0)
    trades = signal.diff().abs().fillna(0)
    strategy_returns = signal.shift(1) * returns - trades * transaction_cost
    return strategy_returns, signal.shift(1)

