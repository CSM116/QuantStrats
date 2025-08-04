

"""
macro_signals.py

Signals adjusted for macroeconomic and seasonal regimes.
Includes macro regime filters (e.g. inflation, PMI) and earnings-season drift signals.
Useful for conditioning traditional signals on fundamental context.
"""


import pandas as pd
import numpy as np


def macro_regime_filter(macro_series, threshold, direction='above'):
    """
    Returns 1 if macro variable is in desired regime, 0 otherwise.
    Used to condition other signals (e.g., only trade momentum in growth).
    """
    if direction == 'above':
        return (macro_series > threshold).astype(int)
    else:
        return (macro_series < threshold).astype(int)



def earnings_drift_signal(earnings_df, days_since_earnings, max_days=40):
    """
    Creates a signal that decays after positive/negative earnings surprise.
    Long for positive surprise; short for negative, decays linearly over `max_days`.
    
    `earnings_df` should have: index = date, columns = ['surprise']
    """
    signal = earnings_df['surprise'].apply(lambda x: 1 if x > 0 else -1)
    signal_df = pd.DataFrame(index=earnings_df.index, data={'signal': signal})
    
    # Expand over days_since_earnings
    out = pd.Series(0, index=days_since_earnings.index)
    for day in range(max_days):
        mask = days_since_earnings == day
        decay = (max_days - day) / max_days
        out[mask] = signal_df['signal'][mask] * decay
    return out


def seasonal_return_pattern(series):
    """
    Computes average return by calendar month.
    Helps detect seasonal strength/weakness tendencies.
    """
    return series.groupby(series.index.month).mean()



def pre_post_fomc_return(series, fomc_dates, window=5):
    """
    Calculates average return N days before and after FOMC dates.
    Used to capture policy-driven price behavior.
    """
    pre_returns, post_returns = [], []
    for date in fomc_dates:
        if date in series.index:
            pre = series.loc[date - pd.Timedelta(days=window):date].pct_change().sum()
            post = series.loc[date:date + pd.Timedelta(days=window)].pct_change().sum()
            pre_returns.append(pre)
            post_returns.append(post)
    return pd.Series(pre_returns), pd.Series(post_returns)
