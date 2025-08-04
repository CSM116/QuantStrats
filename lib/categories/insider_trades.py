import pandas as pd
import numpy as np
from signals_utils import validate_series

def insider_buy_ratio(insider_buys, insider_sells):
    """
    Insider Buy Ratio: insider buys / (insider buys + insider sells).
    Measures net insider sentiment.
    """
    b = validate_series(insider_buys, "insider_buys")
    s = validate_series(insider_sells, "insider_sells")
    total = b + s
    return b / total.replace(0, np.nan)

def net_insider_transactions(insider_buys, insider_sells):
    """
    Net Insider Transactions: buys - sells.
    Positive = net accumulation; negative = distribution.
    """
    b = validate_series(insider_buys, "insider_buys")
    s = validate_series(insider_sells, "insider_sells")
    return b - s

def insider_transaction_value(volume_shares, price):
    """
    Insider Transaction Value: total transaction size in $.
    Useful to weight signals by dollar value.
    """
    v = validate_series(volume_shares, "volume_shares")
    p = validate_series(price, "price")
    return v * p

def recent_insider_buy_activity(insider_buys, window=20):
    """
    Recent Insider Buy Activity: rolling sum of buys.
    Higher values = clustering of insider accumulation.
    """
    b = validate_series(insider_buys, "insider_buys")
    return b.rolling(window).sum()

def insider_activity_zscore(insider_activity, window=60):
    """
    Insider Activity Z-Score: standardizes buy/sell activity.
    Flags abnormal buying or selling pressure.
    """
    x = validate_series(insider_activity, "insider_activity")
    mean = x.rolling(window).mean()
    std = x.rolling(window).std()
    return (x - mean) / std.replace(0, np.nan)
