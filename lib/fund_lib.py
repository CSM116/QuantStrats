

from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def add_forward_returns(df, horizons=[1,2,3,4]):
    df = df.sort_values(['permno', 'dlycaldt'])
    for h in horizons:
        df[f'fr{h}'] = df.groupby('permno')['adj_prc'].pct_change(periods=h, fill_method=None).shift(-h)        
    return df


def forward_fill_metric(df, metrics):
    df = df.sort_values(['permno', 'rdq'])
    df['rdq'] = pd.to_datetime(df['rdq'])
    df[metrics] = df.groupby('permno')[metrics].ffill()
    return df


def filter_midcaps(df, lower_q=0.25, upper_q=0.75, min_total=100):
    if len(df) < min_total:
        print(f"Warning: Only {len(df)} stocks available — using unfiltered universe.")
        return df
    q_low, q_high = df['dlycap'].quantile([lower_q, upper_q])
    return df[(df['dlycap'] >= q_low) & (df['dlycap'] <= q_high)]


def get_monthly_snapshot(df, metrics, month, horizons):
    month_df = df[df['month'] == month]
    cols = ['permno', 'dlycap'] + metrics + [f'fr{h}' for h in horizons]
    return month_df[cols].dropna()



def fast_compute_ics_multi(df, metrics, horizons, min_obs=5):
    cols = metrics + [f'fr{h}' for h in horizons]
    temp = df[cols].dropna()
    if len(temp) < min_obs:
        return {f'IC_fr{h}': [np.nan] * len(metrics) for h in horizons}

    ranked = temp.rank()
    corr_matrix = ranked.corr(method='pearson')

    result = {}
    for h in horizons:
        fr_col = f'fr{h}'
        if fr_col not in corr_matrix.columns:
            result[f'IC_fr{h}'] = [np.nan] * len(metrics)
        else:
            result[f'IC_fr{h}'] = [
                corr_matrix.loc[m, fr_col] if m in corr_matrix.index else np.nan
                for m in metrics
            ]
    return result




def run_rolling_ic_pipeline(df, metrics, horizons=[1, 2, 3, 4]):
    df = add_forward_returns(df, horizons)
    df = forward_fill_metric(df, metrics)

    df['month'] = df['dlycaldt'].dt.to_period('M')
    df = df[df['dlycaldt'] >= df['rdq']]
    unique_months = df['month'].drop_duplicates().sort_values()

    # Initialize results dict: one entry per horizon
    results = {f'IC_fr{h}': [] for h in horizons}

    for month in unique_months:
        snapshot = get_monthly_snapshot(df, metrics, month, horizons)
        if len(snapshot) == 0:
            print(f"{month}: Skipping — no data available.")
            continue

        midcap_df = filter_midcaps(snapshot)
        ic_rows = fast_compute_ics_multi(midcap_df, metrics, horizons)

        for key in ic_rows:
            results[key].append(pd.Series(ic_rows[key], index=metrics, name=month.to_timestamp()))

    # Convert each list of Series into a DataFrame
    for key in results:
        results[key] = pd.DataFrame(results[key])

    return results


def compute_windowed_stability_metrics(ic_series, window):
    ic_series = ic_series.dropna()
    n = len(ic_series)
    n_windows = n // window
    if n_windows < 2:
        return np.nan, np.nan
    
    reshaped = ic_series.iloc[:n_windows * window].values.reshape(-1, window)
    window_means = reshaped.mean(axis=1)
    signs = np.sign(window_means)

    pct_pos = (window_means > 0).mean()
    sign_flips = (np.diff(signs) != 0).sum()

    # Normalize flips to a [0, 1] scale
    norm_sign_flips = sign_flips / (len(signs) - 1)

    return pct_pos, norm_sign_flips



def summarize_ic_stats(results_roll_ic, window=8):
    summary_stats = {h: [] for h in results_roll_ic.keys()}
    rolling_stats = {
        h: {'mean_ic': [], 'std_ic': [], 'ic_sharpe': [], 't_stat': []}
        for h in results_roll_ic.keys()
    }
    
    for h, df_ic in results_roll_ic.items():
        for metric in df_ic.columns:
            ic_series = df_ic[metric].dropna()
    
            mean_ic = ic_series.mean()
            std_ic = ic_series.std()
            ic_sharpe = mean_ic / std_ic if std_ic > 0 else 0
            t_stat = mean_ic / (std_ic / np.sqrt(len(ic_series)))
            
            pct_pos, sign_flips = compute_windowed_stability_metrics(ic_series, window)
            
            summary_stats[h].append({
                'metric': metric,
                'mean_ic': mean_ic,
                'std_ic': std_ic,
                'ic_sharpe': ic_sharpe,
                't_stat': t_stat,
                'pct_pos_windows':pct_pos,
                'sign_flips':sign_flips
            })
    
            # Rolling stats
            rolling_ic_mean = ic_series.rolling(window).mean()
            rolling_ic_std = ic_series.rolling(window).std()
            rolling_ic_sharpe = rolling_ic_mean / rolling_ic_std
            rolling_ic_t_stat = rolling_ic_mean / (rolling_ic_std / np.sqrt(window))
            
            # Store each rolling series with metric as name
            rolling_stats[h]['mean_ic'].append(rolling_ic_mean.rename(metric))
            rolling_stats[h]['std_ic'].append(rolling_ic_std.rename(metric))
            rolling_stats[h]['ic_sharpe'].append(rolling_ic_sharpe.rename(metric))
            rolling_stats[h]['t_stat'].append(rolling_ic_t_stat.rename(metric))
            
        summary_stats[h] = pd.DataFrame(summary_stats[h])
        
    # Convert lists of Series to DataFrames: rows = dates, columns = metrics
    for h in rolling_stats:
        for stat in ['mean_ic', 'std_ic', 'ic_sharpe', 't_stat']:
            rolling_stats[h][stat] = pd.concat(rolling_stats[h][stat], axis=1)
            rolling_stats[h][stat].index = rolling_stats[h][stat].index.strftime('%Y-%m')

    return summary_stats, rolling_stats



def normalize_series(series):
    return (series - series.min()) / (series.max() - series.min())


from scipy.stats import zscore

def compute_rolling_stability_scores(rolling_stats):
    """
    For each horizon, compute a rolling stability score per metric by aggregating
    the std dev across rolling stats.

    Returns:
        Dict[horizon] -> DataFrame(index=metrics, columns=rolling_stability_score + component stds)
    """
    horizon_scores = {}
    stat_keys = ['mean_ic', 'ic_sharpe', 't_stat']

    for h, stat_dict in rolling_stats.items():
        component_stds = {}

        for stat in stat_keys:
            df = stat_dict[stat]  # shape: time × metric
            std_per_metric = df.std(axis=0)  # std over time, for each signal
            component_stds[f'std_{stat}'] = std_per_metric
        score_df = pd.DataFrame(component_stds)
        # Higher std = less stable, so we negate the z-scored sum
        score_df['rolling_stability_score'] = -zscore(score_df, axis=0).sum(axis=1)

        horizon_scores[h] = score_df
        horizon_scores[h].sort_values('rolling_stability_score', ascending=False, inplace=True)
    return horizon_scores


def is_stable_signal(df, sharpe_thresh=0.4, tstat_thresh=2.0, pct_pos_thresh=0.6, sign_flips_thresh=2):
    """
    Returns a boolean mask indicating which rows (signals) are stable.
    """
    return (
        (df['ic_sharpe'] >= sharpe_thresh) &
        (df['t_stat'] >= tstat_thresh) &
        (df['pct_pos_windows'] >= pct_pos_thresh) &
        (df['sign_flips'] <= sign_flips_thresh)
    )


from scipy.stats import zscore

def compute_stability_score(df, var_threshold=1e-8):
    """
    Compute composite stability score using a fixed feature set.
    Low-variance features contribute zero to preserve score comparability.
    """
    score = pd.Series(0.0, index=df.index)

    feature_weights = {
        'ic_sharpe': +1,
        't_stat': +1,
        'pct_pos_windows': +1,
        'sign_flips': -1,
        'std_ic': -1,
    }

    for col, weight in feature_weights.items():
        if col not in df.columns:
            raise ValueError(f"Missing required feature: {col}")

        std = df[col].std()
        if std < var_threshold or df[col].isna().all():
            print(f"⚠️ '{col}' has low variance (std={std:.2e}) — excluded from score.")
            component = pd.Series(0.0, index=df.index)
        else:
            component = zscore(df[col]) * weight

        score += component

    return pd.Series(score, index=df.index, name='stability_score')





def compute_composite_final_score(summary_stats, rolling_stats, comp_score_weight= 0.65, roll_stability_score_weight=0.35):
    rolling_stab_scores = compute_rolling_stability_scores(rolling_stats)
    
    for key, df in summary_stats.items():
        # Ensure 'metric' is index only if not already
        if 'metric' in df.columns:
            df = df.set_index('metric')
    
        if 'stable_signal' not in df.columns:
            df['stable_signal'] = is_stable_signal(df)
    
        # Compute or update stability score if not already present
        if 'stability_score' not in df.columns:
            print(f'key: {key}')
            df['stability_score'] = compute_stability_score(df)
    
        # Join rolling stability score only if not already present
        rolling_df = rolling_stab_scores[key]
        if 'rolling_stability_score' not in df.columns:
            df = df.join(rolling_df[['rolling_stability_score']], how='left')
    
        # Final combined score
        df['final_score'] = (
            comp_score_weight * df['stability_score'] +  # Favor signals that persist across regimes
            roll_stability_score_weight * df['rolling_stability_score']  # Still want to avoid unstable signals
        )
        df.sort_values('final_score', ascending=False, inplace=True)
    
        summary_stats[key] = df
    return summary_stats



##############################  PLOTTING  ################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def heatmap_ic(rolling_stats, horizon=1, param='mean_ic'):
    import os
    os.makedirs("figures", exist_ok=True)

    ic_sharpe_df = rolling_stats[f'IC_fr{horizon}'][param].T

    plt.figure(figsize=(14, 6))

    # Custom colormap and better centering for diverging data
    sns.heatmap(
        ic_sharpe_df,
        cmap='coolwarm',
        center=0,
        cbar_kws={"label": "Rolling Spearman IC"},
        linewidths=0.3,
        linecolor='white'
    )

    # Titles and labels
    plt.title(f"Rolling IC FwdRet horizon: {horizon} Param: {param}", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Metric", fontsize=12)

    # Format x-ticks
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()

    # Save in high resolution
    plt.savefig(f"figures/rolling_heatmap_{param}_fr{horizon}.svg", format="svg")
    plt.show()


    
def plot_ic_series(ic_dictio, horizon=1):
    import matplotlib.dates as mdates
    import seaborn as sns
    import os

    ic_df = ic_dictio[f'IC_fr{horizon}']
    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(12, 6))

    # Use seaborn color palette for cleaner distinction
    palette = sns.color_palette("colorblind", n_colors=len(ic_df.columns))

    for i, col in enumerate(ic_df.columns):
        plt.plot(ic_df.index, ic_df[col], label=col, linewidth=2, color=palette[i])

    # Horizontal line at zero
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    # Format x-axis for better date display
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    # Labels & title
    plt.xlabel('Quarter', fontsize=12)
    plt.ylabel('Spearman IC', fontsize=12)
    plt.title(f'Information Coefficient Over Time (Horizon {horizon})', fontsize=14)

    # Add grid and legend
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(title='Metrics', frameon=False, fontsize=10)
    plt.tight_layout()

    # Save as high-res for publication
    plt.savefig(f"figures/IC_series_FR{horizon}.svg", format="svg")

    plt.show()

    print("Average ICs across all quarters:")
    print(ic_df.mean().round(4))





def plot_rolling_tstat(ic_df, metric, window=8):
    rolling_tstats = {}

    for col in ic_df.columns:
        series = ic_df[col]
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        t_stat = rolling_mean / (rolling_std / np.sqrt(window))
        rolling_tstats[col] = t_stat

    tstat_df = pd.DataFrame(rolling_tstats)

    # Plot
    plt.figure(figsize=(10, 6))
    for col in tstat_df.columns:
        plt.plot(tstat_df.index, tstat_df[col], label=col)

    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.axhline(2, color='green', linestyle='--', linewidth=1, label='t = 2')
    plt.axhline(-2, color='red', linestyle='--', linewidth=1, label='t = -2')
    plt.title(f'{metric} - Rolling {window}-Quarter t-Statistic of IC')
    plt.xlabel('Quarter')
    plt.ylabel('Rolling t-stat')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/rolling_stat_{metric}.svg", format="svg")
    plt.show()

    return tstat_df



import matplotlib.pyplot as plt
import seaborn as sns

def plot_composite_score_heatmap(summary_stats, top_n=2):
    """
    Aggregates and visualizes final composite scores across all horizons.
    
    Parameters:
        summary_stats (dict): Dictionary of DataFrames with 'final_score' per signal and horizon.
        top_n (int): Number of top signals to display in the summary printout.
        
    Returns:
        composite_df (DataFrame): Final score matrix (horizons × metrics).
    """
    import os
    os.makedirs("figures", exist_ok=True)
    
    # Step 1: Build and sort score matrix
    composite_scores_df = {
        h: horizon_df[['final_score']].squeeze()
        for h, horizon_df in summary_stats.items()
    }
    composite_df = pd.DataFrame(composite_scores_df).T
    composite_df = composite_df[composite_df.mean().sort_values(ascending=False).index]
    
    # Step 2: Plot
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        composite_df,
        annot=True,
        fmt=".2f",
        cmap="rocket_r",
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Final Score'}
    )
    
    # Step 3: Improve labels and layout
    ax.set_title("Final Composite Signal Score per Horizon", fontsize=14)
    ax.set_xlabel("Signal (Metric)", fontsize=12)
    ax.set_ylabel("Forward Return Horizon", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    # plt.savefig("figures/composite_score_heatmap.png", dpi=300)
    plt.savefig("figures/composite_score_heatmap.svg", format="svg")
    plt.show()
    
    # Step 4: Print top signals
    mean_scores = composite_df.mean(axis=0).sort_values(ascending=False)
    print(f"\nTop {top_n} signals by mean final score:")
    print(mean_scores.head(top_n).round(4))
    
    return composite_df

