import pandas as pd
import numpy as np
from signals_utils import validate_series

def analyst_upgrades(upgrades):
    """
    Analyst Upgrades: count or score of upgrade events.
    Indicates improving sentiment.
    """
    return validate_series(upgrades, "upgrades")

def analyst_downgrades(downgrades):
    """
    Analyst Downgrades: count or score of downgrade events.
    Useful as negative signal or reversal trigger.
    """
    return validate_series(downgrades, "downgrades")

def net_rating_changes(upgrades, downgrades):
    """
    Net Rating Changes: upgrades - downgrades.
    Captures net analyst sentiment shift.
    """
    u = validate_series(upgrades, "upgrades")
    d = validate_series(downgrades, "downgrades")
    return u - d

def coverage_initiations(initiations):
    """
    Coverage Initiations: count of new analyst initiations.
    Often lead to sustained attention and price moves.
    """
    return validate_series(initiations, "initiations")

def attention_zscore(attention_series, window=60):
    """
    Analyst Attention Z-Score: standardizes attention-related events.
    Flags unusual levels of analyst activity.
    """
    x = validate_series(attention_series, "attention_series")
    mean = x.rolling(window).mean()
    std = x.rolling(window).std()
    return (x - mean) / std.replace(0, np.nan)
