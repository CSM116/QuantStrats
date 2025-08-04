import pandas as pd
import numpy as np
from signals_utils import validate_series

def eps_surprise(actual_eps, expected_eps):
    """
    EPS Surprise (%): (actual - expected) / |expected|.
    Captures deviation from consensus forecast.
    """
    a = validate_series(actual_eps, "actual_eps")
    e = validate_series(expected_eps, "expected_eps")
    return (a - e) / e.replace(0, np.nan)

def forecast_dispersion(std_estimates, mean_estimates):
    """
    Forecast Dispersion: std / mean of EPS estimates.
    Measures analyst disagreement; higher = more uncertainty.
    """
    std = validate_series(std_estimates, "std_estimates")
    mean = validate_series(mean_estimates, "mean_estimates")
    return std / mean.replace(0, np.nan)

def eps_revision(current_eps, prior_eps):
    """
    EPS Revision (%): (current - prior) / |prior|.
    Tracks upgrades/downgrades over time.
    """
    c = validate_series(current_eps, "current_eps")
    p = validate_series(prior_eps, "prior_eps")
    return (c - p) / p.replace(0, np.nan)

def rolling_eps_revision(eps_series, window=20):
    """
    Rolling EPS Revision: current EPS vs rolling average of past EPS.
    Highlights persistent upgrade or downgrade trends.
    """
    eps = validate_series(eps_series, "eps_series")
    return eps / eps.rolling(window).mean().replace(0, np.nan) - 1

def earnings_surprise_zscore(surprise_series, window=60):
    """
    Earnings Surprise Z-Score: standardizes surprises over history.
    Flags unusually strong or weak earnings beats/misses.
    """
    s = validate_series(surprise_series, "surprise_series")
    mean = s.rolling(window).mean()
    std = s.rolling(window).std()
    return (s - mean) / std.replace(0, np.nan)
