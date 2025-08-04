"""
cross_sectional_signals.py

Cross-sectional signals for relative strength, ranking, and dispersion across equities.
Useful for strategies that select or rank stocks based on relative features.
"""

import pandas as pd
import numpy as np


def zscore_cross_section(df):
    """
    Computes cross-sectional z-score of a DataFrame (each row = time, cols = assets).
    Highlights overbought/oversold names relative to universe.
    """
    return (df - df.mean(axis=1, keepdims=True)) / df.std(axis=1, ddof=0, keepdims=True)


def rank_signal(df, ascending=True):
    """
    Ranks assets at each timestamp.
    Lower ranks (e.g., 1) = top names if ascending=False.
    """
    return df.rank(axis=1, method='first', ascending=ascending)


def dispersion_score(df):
    """
    Calculates cross-sectional return dispersion at each timestamp.
    Higher dispersion = greater alpha potential for long-short strategies.
    """
    return df.std(axis=1, ddof=0)
