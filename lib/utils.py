
"""
signal_explainability.py

This module provides functions to evaluate the statistical relationship between a signal
and future returns across a universe of assets. It helps determine the predictive power
(alpha potential) of signals using standard statistical tests and metrics.
"""


import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, spearmanr, pearsonr
from sklearn.feature_selection import mutual_info_regression
from typing import Union, List


def compute_zscores(price: pd.Series, windows: Union[int, List[int]]) -> Union[pd.Series, pd.DataFrame]:
    """
    Return z-score(s) of *metric* for the given look-back window(s).

    If *windows* is an int, return a single z-score Series.
    If *windows* is a list, return a DataFrame with one column per window.
    """
    if isinstance(windows, int):
        windows = [windows]
    
    return pd.DataFrame({
        f'z{w}': (price - price.rolling(w).mean()) / price.rolling(w).std()
        for w in windows
    })
    

def compute_forward_returns(price: pd.Series, horizons: Union[int, List[int]]) -> Union[pd.Series, dict]:
    """
    Return forward percentage return(s) for each horizon.

    If *horizons* is an int, return a single Series.
    If *horizons* is a list, return a dict of Series keyed 'fr{h}'.
    where *h* is the horizon length in periods.
    """
    if isinstance(horizons, int):
        horizons = [horizons]
    
    return pd.DataFrame({
        f'fr{h}': price.pct_change(periods=h, fill_method=None).shift(-h)
        for h in horizons
    })


def fast_rolling_spearman(signals: pd.DataFrame, future_returns: pd.Series, window: int) -> pd.DataFrame:
    """
    Compute rolling Spearman IC between each signal column and *future_returns*.

    Rank-transforms both series, then applies rolling Pearson over *window*.
    """
    ranked_signals = signals.rank(method="average")
    ranked_returns = future_returns.rank(method="average")
    rolling_corr = ranked_signals.rolling(window).corr(ranked_returns)
    
    # Rename columns: 'z20' → 'z20_ic'
    rolling_corr.columns = [f"{col}_ic" for col in rolling_corr.columns]
    return rolling_corr


from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

def summarize_ic_stats(correlations_df: pd.DataFrame, lags: int = None, use_newey_west: bool = False) -> pd.DataFrame:
    """
    Compute IC summary statistics with optional Newey-West t-stat adjustment.

    Parameters:
    - correlations_df: pd.DataFrame of ICs
    - lags: int, optional; if set, uses Newey-West with this lag
    - use_newey_west: bool, optional; if True, uses NW even if lags is None (default: False)
    """
    records = []
    for col in correlations_df.columns:
        x = correlations_df[col].dropna()
        n = len(x)
        mean_ic = x.mean()
        std_ic = x.std()
        naive_t = mean_ic / (std_ic / np.sqrt(n))

        if use_newey_west or lags is not None:
            lags_used = int(4 * (n / 100) ** (2 / 9)) if lags is None else lags
            model = OLS(x, np.ones_like(x))
            results = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags_used})
            nw_t = results.tvalues.iloc[0]
        else:
            nw_t = np.nan

        records.append({
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "t_stat_naive": naive_t,
            "t_stat_nw": nw_t,
            "n_obs": n,
            "zscore": col
        })

    return pd.DataFrame(records)



def optimize_zscore_window(price: pd.Series, z_windows: list, horizons: list, ic_window: int = 252,
                          lags: int = None, use_newey_west: bool = False) -> pd.DataFrame:
    """
    Full pipeline:

    1. Compute z-scores across *z_windows*.
    2. Compute forward returns across *horizons*.
    3. For each horizon, calculate rolling Spearman IC over *ic_window*.
    4. Return a long-form DataFrame of IC summary stats sorted by t-stat.

    Output columns:
        ['mean_ic', 'std_ic', 't_stat_naive', 't_stat_nw', 'n_obs', 'horizon', 'zscore']
    """
    zscores = compute_zscores(price, z_windows)
    forward_returns = compute_forward_returns(price, horizons)

    records = []
    for h in horizons:
        fr = forward_returns[f"fr{h}"]
        ic_df = fast_rolling_spearman(zscores, fr, window=ic_window)

        # from statsmodels.graphics.tsaplots import plot_acf
        # ic_series = ic_df['z20'].dropna()
        # plot_acf(ic_series, lags=100, title=f'Autocorrelation - horizon:{h}')

        stats = summarize_ic_stats(ic_df, lags, use_newey_west)
        stats["zscore"] = stats["zscore"].str.replace("_ic$", "", regex=True)

        # add identifiers before storing
        stats["horizon"] = h
        records.append(stats.reset_index(drop=True))

    summary = pd.concat(records, ignore_index=True)
    
    # Average stats across horizons for each z-score window
    # avg_stats = (
    #     summary.groupby("zscore")[["mean_ic", "std_ic", "t_stat_naive","t_stat_nw"]]
    #     .mean()
    #     .sort_values("t_stat_nw")
    #     .reset_index()
    # )
    # display(avg_stats)
    
    return summary


################## PARALLEL PROCESSING ##############

from concurrent.futures import ThreadPoolExecutor, as_completed

def process_single_ticker(symbol, series, z_windows, horizons, ic_window, min_obs, lags, use_newey_west):
    if series.dropna().size < min_obs:
        return None

    try:
        result = optimize_zscore_window(series, z_windows, horizons, ic_window, lags, use_newey_west)
        result["symbol"] = symbol
        return result
    except Exception as e:
        # print(f"Error with {symbol}: {e}")
        return None

def run_parallel_zscore_optimization(prices, z_windows, horizons, ic_window=252, min_obs=500, max_workers=8,
                                    lags: int = None, use_newey_west: bool = False):
    futures = []
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for symbol in prices.columns:
            series = prices[symbol]
            futures.append(
                executor.submit(process_single_ticker, symbol, series, z_windows, horizons, ic_window, min_obs, lags, use_newey_west)
            )

        for f in as_completed(futures):
            res = f.result()
            if res is not None:
                results.append(res)

    return pd.concat(results, ignore_index=True)


############### ANALYZE IC SUMMARY ###############
import seaborn as sns
import matplotlib.pyplot as plt


def analyze_ic_summary(universe_summary: pd.DataFrame, stat_col: str = "t_stat_naive") -> pd.DataFrame:
    """
    Analyze IC summary with optional choice of t-stat column.
    
    Parameters:
    - universe_summary: DataFrame with columns ['zscore', 'horizon', 'mean_ic', 'std_ic', stat_col, ...]
    - stat_col: Which t-stat column to use ('t_stat_naive' or 't_stat_nw')

    Returns:
    - summary_table: Aggregated stats over zscore windows for horizons 20–60
    """

    # Avg t-stat per z-score
    avg_zscore = (
        universe_summary.groupby("zscore")[stat_col]
        .mean()
        .sort_values(ascending=False)
    )

    # Absolute t-stat for ranking
    universe_summary["abs_t_stat"] = universe_summary[stat_col].abs()

    # Pivot heatmap table
    pivot = universe_summary.pivot_table(
        index="zscore",
        columns="horizon",
        values=stat_col,
        aggfunc="median"
    )

    # Sort index numerically by z-score
    pivot = pivot.sort_index(key=lambda x: x.map(lambda s: int(s[1:])))
    
    # Plot
    sns.heatmap(
        pivot,
        cmap="viridis_r",         # inverted color scale
        annot=True,               # show values in each cell
        fmt=".1f",                # number format
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": f"Avg {stat_col}"}
    )
    
    plt.xlabel("Horizon")
    plt.ylabel("Z-score Window")
    plt.title(f"Avg {stat_col} by Z-score Window and Horizon")
    plt.show()

    # Summary table
    summary_table = (
        universe_summary
        .loc[universe_summary["horizon"].between(20, 60)]
        .groupby("zscore")
        .agg(
            mean_ic=("mean_ic", "mean"),
            std_ic=("std_ic", "mean"),
            avg_t_stat=(stat_col, "mean"),
            abs_t_stat=(stat_col, lambda x: np.mean(np.abs(x))),
            n_obs=("n_obs", "sum")
        )
        .sort_values("abs_t_stat", ascending=False)
    )

    return summary_table, pivot



########### STABILITY TEST #########

from datetime import timedelta

def evaluate_zscore_stability_over_time(
    prices: pd.DataFrame,
    z_windows: list,
    horizons: list,
    ic_window: int = 252,
    min_obs: int = 500,
    step_years: int = 1,
    window_years: int = 2,
    max_workers: int = 4,
    lags: int = None, 
    use_newey_west: bool = False,
    stat_col: str = "t_stat_naive"
    ) -> pd.DataFrame:
    """
    Evaluate stability of z-score windows over rolling time periods.

    Returns:
        Long-form DataFrame with:
        - 'start_date'
        - 'zscore'
        - 'horizon'
        - 'mean_ic', 'std_ic', 't_stat_naive'
    """
    results = []
    date_index = prices.index.dropna().sort_values()
    start_date = date_index.min()
    end_date = date_index.max()

    current = start_date
    while current + timedelta(days=365 * window_years) <= end_date:
        window_start = current
        window_end = current + timedelta(days=365 * window_years)
        current += timedelta(days=365 * step_years)

        # Slice price data to this time window
        price_window = prices.loc[(prices.index >= window_start) & (prices.index < window_end)]

        if price_window.dropna(how='all', axis=1).shape[1] == 0:
            continue  # no valid tickers

        try:
            universe_summary = run_parallel_zscore_optimization(
                price_window,
                z_windows,
                horizons,
                ic_window=ic_window,
                min_obs=min_obs,
                max_workers=max_workers,
                lags=lags,
                use_newey_west=use_newey_west
            )

            grouped = (
                universe_summary
                .groupby(["zscore", "horizon"])
                .agg(
                    mean_ic=("mean_ic", "mean"),
                    std_ic=("std_ic", "mean"),
                    t_stat=(stat_col, "mean")
                )
                .reset_index()
            )
            grouped["start_date"] = window_start.strftime("%Y-%m-%d")
            results.append(grouped)
        except Exception as e:
            print(f"Window {window_start.strftime('%Y')} failed: {e}")
            continue

    stability_df = pd.concat(results, ignore_index=True)

    pivot = stability_df.pivot_table(index="start_date", columns="zscore", values="t_stat")

    pivot.plot(figsize=(10, 5))

    plt.ylabel("Avg t-stat")
    plt.title("Z-score Window Stability Over Time")
    
    # Force at least 10 x-ticks
    xticks_locs = np.linspace(0, len(pivot.index) - 1, 10, dtype=int)
    xticks_labels = pivot.index[xticks_locs]
    plt.xticks(xticks_locs, xticks_labels, rotation=45)
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return stability_df












def _validate_series(series, name):
    if not isinstance(series, pd.Series):
        raise ValueError(f"{name} must be a pandas Series.")
    if series.isnull().all():
        raise ValueError(f"{name} contains only NaNs.")
    return series.astype(float)
    


def information_coefficient(signal: pd.Series, future_returns: pd.Series, method: str = 'spearman') -> float:
    """
    Computes the IC (Spearman or Pearson correlation) between signal and future returns.
    """
    valid = signal.dropna().index.intersection(future_returns.dropna().index)
    if len(valid) < 5:
        return float('nan')

    x = signal.loc[valid]
    y = future_returns.loc[valid]

    if method == 'spearman':
        return spearmanr(x, y).correlation
    elif method == 'pearson':
        return pearsonr(x, y)[0]
    elif method == 'kendall':
        return kendalltau(x, y)[0]
    else:
        raise ValueError(f"Unsupported method: {method}")


def quantile_return_spread(signal: pd.Series, future_returns: pd.Series, q: float = 0.2) -> float:
    """
    Computes average return spread between top and bottom quantiles of the signal.
    """
    valid = signal.dropna().index.intersection(future_returns.dropna().index)
    signal, future_returns = signal.loc[valid], future_returns.loc[valid]
    q_low = signal.quantile(q)
    q_high = signal.quantile(1 - q)

    low_group = future_returns[signal <= q_low]
    high_group = future_returns[signal >= q_high]

    return high_group.mean() - low_group.mean()


def conditional_t_stat(signal: pd.Series, future_returns: pd.Series, threshold: float = 1.5) -> float:
    """
    Performs a t-test on future returns where signal is strong (>|threshold|).
    """
    valid = signal.dropna().index.intersection(future_returns.dropna().index)
    signal, future_returns = signal.loc[valid], future_returns.loc[valid]

    selected = future_returns[(signal.abs() > threshold)]
    others = future_returns[(signal.abs() <= threshold)]

    if len(selected) < 5 or len(others) < 5:
        return np.nan
    t_stat, _ = ttest_ind(selected, others, equal_var=False)
    return t_stat


def mutual_info_score(signal: pd.Series, future_returns: pd.Series) -> float:
    """
    Estimates mutual information between signal and future returns.
    """
    valid = signal.dropna().index.intersection(future_returns.dropna().index)
    if len(valid) < 10:
        return np.nan
    return mutual_info_regression(signal.loc[valid].values.reshape(-1, 1), future_returns.loc[valid].values, random_state=42)[0]


def evaluate_signal_panel(signals: pd.DataFrame, prices: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Evaluates a signal across assets using multiple statistical tests.
    Returns a DataFrame of results (rows = tickers).
    """
    future_rets = compute_forward_returns(prices, horizon)
    results = []

    for ticker in signals.columns:
        s = signals[ticker]
        r = future_rets[ticker]
        results.append({
            'Ticker': ticker,
            'IC (Spearman)': information_coefficient(s, r, 'spearman'),
            'IC (Pearson)': information_coefficient(s, r, 'pearson'),
            'Return Spread (20%)': quantile_return_spread(s, r),
            'T-stat': conditional_t_stat(s, r),
            'Mutual Info': mutual_info_score(s, r)
        })

    return pd.DataFrame(results).set_index('Ticker')


def test_normality(series, method="shapiro", alpha=0.05):
    """
    Tests for normality of a series.
    - method: 'shapiro', 'anderson', or 'jarque_bera'
    Returns test statistic and p-value (or significance result).
    """
    from scipy import stats

    if method == "shapiro":
        stat, p = stats.shapiro(series.dropna())
        result = p > alpha
    elif method == "anderson":
        stat = stats.anderson(series.dropna()).statistic
        result = None  # no p-value
    elif method == "jarque_bera":
        stat, p = stats.jarque_bera(series.dropna())
        result = p > alpha
    else:
        raise ValueError("Invalid method")

    return {"statistic": stat, "p_value": p if method != "anderson" else None, "normal": result}



def rolling_correlation_with_return(signals: pd.DataFrame, future_returns: pd.Series, window: int = 60, method: str = 'spearman') -> pd.DataFrame:
    """
    Computes rolling correlation (IC) between each signal and future returns.

    Returns:
    - DataFrame of rolling ICs, shape [T, N]
    """
    results = pd.DataFrame(index=signals.index, columns=signals.columns, dtype=float)

    for col in signals.columns:
        for i in range(window - 1, len(signals)):
            x_window = signals[col].iloc[i - window + 1: i + 1]
            y_window = future_returns.iloc[i - window + 1: i + 1]
            ic = information_coefficient(x_window, y_window, method=method)
            results.at[signals.index[i], col] = ic

    return results






