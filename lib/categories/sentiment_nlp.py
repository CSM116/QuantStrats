import pandas as pd
import numpy as np
from signals_utils import validate_series

def news_sentiment_score(sentiment_series):
    """
    News Sentiment Score: average of sentiment scores over time.
    Positive values = bullish tone; negative = bearish.
    """
    return validate_series(sentiment_series, "sentiment_series")

def sentiment_zscore(sentiment_series, window=60):
    """
    Sentiment Z-Score: standardizes sentiment to flag extreme views.
    High = unusually positive; Low = unusually negative.
    """
    s = validate_series(sentiment_series, "sentiment_series")
    mean = s.rolling(window).mean()
    std = s.rolling(window).std()
    return (s - mean) / std.replace(0, np.nan)

def sentiment_diff(current_sentiment, prev_sentiment):
    """
    Sentiment Diff: current - previous sentiment.
    Captures changes in tone or direction.
    """
    c = validate_series(current_sentiment, "current_sentiment")
    p = validate_series(prev_sentiment, "prev_sentiment")
    return c - p

def social_sentiment_score(likes, retweets, mentions):
    """
    Social Sentiment Score: simple engagement-weighted metric.
    Proxy for retail interest or attention.
    """
    l = validate_series(likes, "likes")
    r = validate_series(retweets, "retweets")
    m = validate_series(mentions, "mentions")
    return (l + 2 * r + 0.5 * m)  # Weights are heuristic, can be tuned

def sentiment_spike(sentiment_series, window=20):
    """
    Sentiment Spike: sentiment / rolling average.
    Flags abnormal sentiment bursts.
    """
    s = validate_series(sentiment_series, "sentiment_series")
    avg = s.rolling(window).mean()
    return s / avg.replace(0, np.nan)