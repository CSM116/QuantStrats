import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

div_zero_const = 1e-10 # Avoid division by zero


def find_missing_trading_days(df):
    """
    Identifies missing trading days (excluding weekends) in a DataFrame indexed by DatetimeIndex.

    Parameters:
    - df: Pandas DataFrame with a DatetimeIndex.

    Returns:
    - List of missing trading days (excluding weekends).
    """
    # Ensure index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    
    # Generate full expected trading days (weekdays only)
    expected_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')  # 'B' excludes weekends

    # Find missing trading days
    missing_dates = expected_dates.difference(df.index)

    return missing_dates


def compute_rsi(prices, window=20): # 20 trading days (1 month)
    """
    Computes RSI from price differences.
    
    Parameters:
    prices : pd.DataFrame
        DataFrame of asset prices.
    window : int, default=20
        Lookback period for RSI.
    
    Returns:
    pd.DataFrame
        RSI values for each asset.
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).ewm(span=window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=window, adjust=False).mean()
    rs = gain / (loss + div_zero_const)
    return 100 - (100 / (1 + rs))



def compute_features(prices):
    """
    Computes a feature map based on historical price data.
    
    Parameters:
    prices : pd.DataFrame
        DataFrame of asset prices with columns as assets and rows as dates.
    
    Returns:
    faetures : dict
        Feature map with containing:
            r_1 :  1-day lagged return
            r_5 :  5-day cumulative return
            r_10 : 10-day cumulative return
            z_10 : Rolling Z-score of Returns (10-day)
            vol_10 : 10-day rolling standard deviation
            rv_10 :  10-day realized volatility
            vol_q_10 :  Robust volatility
            rsi_20 :  relative sttrenght index (20-day)
            sma_ratio :  Ratio for short-term trend 5day/20day
            bb_pct : Bollinger Bands %B
            ewma_5 : 5-day Exponentially Weighted Moving Average
            ewma_20 : 20-day Exponentially Weighted Moving Average
            macd : MACD (Moving Average Convergence Divergence) 10-day - 30-day
            auto_corr_5 : 5-day autocorrelation
    """

    # Daily returns
    returns = prices.pct_change()
    
    # Momentum Features
    r_1 = returns.shift(1)  # 1-day lagged return
    r_5 = returns.rolling(5).sum()  # 5-day cumulative return
    r_10 = returns.rolling(10).sum()  # 10-day cumulative return

    # Rolling Z-score of Returns (10-day)
    rolling_mean = returns.rolling(10).mean().shift(1)
    rolling_std = returns.rolling(10).std(ddof=0).shift(1).replace(0, np.nan)
    z_10 = (returns - rolling_mean) / rolling_std  

    # Volatility Features
    vol_10 = returns.rolling(10).std(ddof=0)  # 10-day rolling standard deviation
    rv_10 = (returns ** 2).rolling(10).sum()  # 10-day realized volatility
    vol_q_10 = returns.rolling(10).quantile(0.75) - returns.rolling(10).quantile(0.25)  # Robust volatility

    # RSI (20-day)
    rsi_20 = compute_rsi(prices)

    # Moving Averages
    sma_5 = prices.rolling(5).mean()
    sma_20 = prices.rolling(20).mean()
    sma_ratio = sma_5 / sma_20.replace(0, np.nan)  # Ratio for short-term trend

    # Exponentially Weighted Moving Averages (EWMA)
    ewma_5 = returns.ewm(span=5).mean()
    ewma_20 = returns.ewm(span=20).mean()

    # MACD (Moving Average Convergence Divergence)
    ewma_10 = returns.ewm(span=10).mean()
    ewma_30 = returns.ewm(span=30).mean()
    macd = ewma_10 - ewma_30
    
    # Bollinger Bands %B
    rolling_std_20 = prices.rolling(20).std(ddof=0).replace(0, np.nan)
    upper_band = sma_20 + 2 * rolling_std_20
    lower_band = sma_20 - 2 * rolling_std_20
    bb_pct = (prices - lower_band) / (upper_band - lower_band).replace(0, np.nan)

    # Autocorrelation of 5-day Returns
    auto_corr_5 = returns.rolling(5).apply(lambda x: x.autocorr(), raw=False)

    features = {"r_1":r_1,"r_5":r_5,"r_10":r_10,"z_10":z_10,"vol_10":vol_10,
                "rv_10":rv_10,"vol_q_10":vol_q_10,"rsi_20":rsi_20,"sma_ratio":sma_ratio,
                "bb_pct":bb_pct,"ewma_5":ewma_5,"ewma_20":ewma_20,"macd":macd,"auto_corr_5":auto_corr_5}

    return features

def generate_long_ftr_map(features, target_return):    
    """
    Efficiently stack all DataFrames, assign feature names, and concatenate
    # Combine Features into a Single DataFrame

    Parameters:
    features : dict
        Dictionary of extracted fetures.
    target_return : pd.Dataframe
        Dataframe of target returns

    Output:
    features_map : pd.Dataframe multiindex
        Long format of feature map
    target_rets_final: pd.Dataframe
        Target/Expected output returns
    """ 
    
    features_map = (
        pd.concat([df.stack() for df in features.values()], axis=1)
        .reset_index()
        .rename(columns={"level_1": "Asset"})
        .sort_values(by=["date","Asset"])  # Ensure sorting by Date
        .set_index("date")  # Set Date as index
    )
    
    # Assign column names
    features_map.columns = ["Asset"] + list(features.keys())
    
    # Reshape the target
    target_rets_final = target_return.melt(ignore_index=False, var_name="Asset", value_name="Target")
    target_rets_final = target_rets_final.sort_values(by=["date", "Asset"]).rename_axis("date")

    return features_map, target_rets_final

# pdb.set_trace()

def plot_strategy_performance(cumulative_returns, signal, pnl, rolling_sharpe):
    figwidth = 12
    figheight = 2

    # Plot Performance Metrics
    plt.figure(figsize=(figwidth, figheight))
    plt.plot(cumulative_returns, label="Strategy Cumulative Return")
    plt.axhline(1, color="black", linestyle="--", linewidth=0.8)
    plt.title("Cumulative Return of Strategy")
    plt.legend()
    plt.show()

    # Plot Positioning
    plt.figure(figsize=(figwidth, figheight))
    plt.plot(signal.sum(axis=1), label="Net Exposure ($)", color='red')
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.title("Net Positioning Over Time")
    plt.legend()
    plt.show()
    
    # Plot Return Distribution
    plt.figure(figsize=(figwidth, figheight))
    plt.hist(pnl, bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.axvline(pnl.mean(), color='red', linestyle='dashed', linewidth=2, label="Mean Return")
    plt.ylabel("Frequency")
    plt.xlabel("Trade PnL")
    plt.legend()
    plt.grid()
    plt.show()

    # Compute and plot drawdown
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns - peak
    plt.figure(figsize=(figwidth, figheight))
    plt.fill_between(cumulative_returns.index, drawdown, 0, color="red", alpha=0.5)
    plt.title("Drawdown Curve")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.show()
    
    # Plot rolling conditional Sharpe ratio
    plt.figure(figsize=(figwidth, figheight))
    plt.plot(rolling_sharpe, label="Rolling Conditional Sharpe", color='green')
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.title("Rolling Conditional Sharpe Ratio")
    plt.legend()
    plt.grid(True)
    plt.show()
    

