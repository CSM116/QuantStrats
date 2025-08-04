
import pandas as pd
import numpy as np



def adx(high, low, close, window=14):
    """
    Computes the Average Directional Index (ADX), a trend strength indicator.
    Values > 25 suggest strong trend; < 20 suggest weak or sideways movement.
    """
    # [Implementation here]

def percentile_rank(series, window=60):
    """
    Returns the rolling percentile rank of the most recent value.
    Useful for identifying extremes (e.g. top/bottom 10% of price range).
    """
    return series.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])


def rsi_divergence(series, rsi_window=14, lookback=20):
    """
    Detects divergence between price and RSI direction.
    Flags potential bullish (1) or bearish (-1) divergences.
    
    Parameters:
        series (pd.Series): Price series.
        rsi_window (int): RSI calculation period.
        lookback (int): Number of bars to scan for peaks/troughs.

    Returns:
        pd.Series: Series with 1 for bullish divergence, -1 for bearish, 0 otherwise.
    """
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=rsi_window).mean()
    avg_loss = pd.Series(loss).rolling(window=rsi_window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = pd.Series(rsi, index=series.index)

    signal = pd.Series(0, index=series.index)

    for i in range(lookback, len(series)):
        price_window = series[i - lookback:i]
        rsi_window_vals = rsi[i - lookback:i]

        # Find local price lows and highs
        price_min_idx = price_window.idxmin()
        price_max_idx = price_window.idxmax()

        # Corresponding RSI values
        rsi_at_min = rsi.loc[price_min_idx]
        rsi_at_max = rsi.loc[price_max_idx]

        # Bullish divergence: price makes lower low, RSI makes higher low
        if (series[i] < series[price_min_idx]) and (rsi[i] > rsi_at_min):
            signal[i] = 1
        # Bearish divergence: price makes higher high, RSI makes lower high
        elif (series[i] > series[price_max_idx]) and (rsi[i] < rsi_at_max):
            signal[i] = -1

    return signal
