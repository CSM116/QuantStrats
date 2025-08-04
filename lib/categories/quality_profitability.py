import pandas as pd
import numpy as np
from signals_utils import validate_series

def return_on_invested_capital(nopat, invested_capital):
    """
    ROIC: NOPAT / Invested Capital.
    Measures efficiency of capital use; higher is better.
    """
    n = validate_series(nopat, "nopat")
    ic = validate_series(invested_capital, "invested_capital")
    return n / ic.replace(0, np.nan)

def gross_margin(gross_profit, revenue):
    """
    Gross Margin: gross profit / revenue.
    Indicates product-level profitability.
    """
    gp = validate_series(gross_profit, "gross_profit")
    r = validate_series(revenue, "revenue")
    return gp / r.replace(0, np.nan)

def operating_margin(operating_income, revenue):
    """
    Operating Margin: operating income / revenue.
    Reflects core business profitability.
    """
    oi = validate_series(operating_income, "operating_income")
    r = validate_series(revenue, "revenue")
    return oi / r.replace(0, np.nan)

def free_cash_flow_yield(free_cash_flow, market_cap):
    """
    FCF Yield: free cash flow / market cap.
    Cash return relative to valuation.
    """
    fcf = validate_series(free_cash_flow, "free_cash_flow")
    mc = validate_series(market_cap, "market_cap")
    return fcf / mc.replace(0, np.nan)

def net_margin(net_income, revenue):
    """
    Net Margin: net income / revenue.
    Final profitability after all expenses.
    """
    ni = validate_series(net_income, "net_income")
    r = validate_series(revenue, "revenue")
    return ni / r.replace(0, np.nan)

def quality_zscore(quality_metric, window=60):
    """
    Quality Z-Score: standardizes profitability vs historical levels.
    """
    q = validate_series(quality_metric, "quality_metric")
    mean = q.rolling(window).mean()
    std = q.rolling(window).std()
    return (q - mean) / std.replace(0, np.nan)
