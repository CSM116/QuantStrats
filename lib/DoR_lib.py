import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from datetime import date

import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

def add_return_features(df):
    adj_close = df.get('Adj Close', df['Close'])
    df = df.assign(
        **{
            'C-C Return': (adj_close - adj_close.shift(1)) / adj_close.shift(1),
            'H-L Return': (df['High'] - df['Low']) / df['Low'],
            'O-C Return': (df['Open'] - df['Close']) / df['Open'],
            'True Range': pd.concat([
                df['High'] - df['Low'],
                (df['High'] - df['Close'].shift(1)).abs(),
                (df['Low'] - df['Close'].shift(1)).abs()
            ], axis=1).max(axis=1) / df['Open']
        }
    )
    return df

def add_return_features_multistock(df):
    # 2. Clean per ticker
    for ticker in df.columns.levels[0]:   
        # 3. Generate return features per ticker
        try:
            df[(ticker, 'C-C Return')] = (df[(ticker, 'Adj Close')] - df[(ticker, 'Adj Close')].shift(1)) / df[(ticker, 'Adj Close')].shift(1)
        except:
            df[(ticker, 'C-C Return')] = (df[(ticker, 'Close')] - df[(ticker, 'Close')].shift(1)) / df[(ticker, 'Close')].shift(1)
    
        df[(ticker, 'H-L Return')] = (df[(ticker, 'High')] - df[(ticker, 'Low')]) / df[(ticker, 'Low')]
        df[(ticker, 'O-C Return')] = (df[(ticker, 'Open')] - df[(ticker, 'Close')]) / df[(ticker, 'Open')]
    
        diff_h_l = df[(ticker, 'High')] - df[(ticker, 'Low')]
        diff_h_c = abs(df[(ticker, 'High')] - df[(ticker, 'Close')].shift(1))
        diff_l_c = abs(df[(ticker, 'Low')] - df[(ticker, 'Close')].shift(1))
    
        df[(ticker, 'True Range')] = pd.concat([diff_h_l, diff_h_c, diff_l_c], axis=1).max(axis=1) / df[(ticker, 'Open')]
       
    return df


    
def download_and_clean(symbol, from_date, to_date, freq, singleStock=True, max_retries=3, retry_delay=5):
    attempt = 0
    while attempt < max_retries:
        try:
            df = yf.download(symbol, start=from_date, end=to_date, group_by='ticker',
                             auto_adjust=False, interval=freq, progress=False, threads=True)
            break  # success
        except Exception as e:
            print(f"Download error for {symbol} (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(retry_delay)
            attempt += 1
    else:
        print(f"Failed to download data for {symbol} after {max_retries} attempts.")
        return pd.DataFrame()  # or raise an error

    if df.empty:
        print(f"No data for {symbol}")
        return df

    df.sort_index(ascending=True, inplace=True)

    if freq == '3mo':
        df = df.dropna()

    if singleStock:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
        df.columns.name = None

        mask = (df['Open'] == df['High']) & (df['High'] == df['Low']) & (df['Low'] == df['Close'])
        df.loc[mask, :] = np.nan

        df['Open'] = df['Open'].replace(0, np.nan).ffill(limit=1)

        df = add_return_features(df)

    else:
        for ticker in df.columns.levels[0]:
            mask = (
                (df[(ticker, 'Open')] == df[(ticker, 'High')]) &
                (df[(ticker, 'High')] == df[(ticker, 'Low')]) &
                (df[(ticker, 'Low')] == df[(ticker, 'Close')])
            )
            df.loc[mask, (ticker, slice(None))] = np.nan

            df[(ticker, 'Open')] = df[(ticker, 'Open')].replace(0, np.nan).ffill(limit=1)

        df = add_return_features_multistock(df)

    df.index = pd.Index(df.index.date, name='Date')
    return df




def analyze_returns(df, return_col='C-C Return', std_levels=[1, 2, 3]):
    """
    Combines return sign summary and standard deviation coverage into one function.
    Returns:
        posneg_df: Summary of positive/negative/zero returns
        std_df: Coverage within N std deviations
        stats: Mean and std of the return column
    """
    mean_ret = df[return_col].mean()
    std_ret = df[return_col].std()
    total = len(df)

    # Sign summary
    masks = [df[return_col] > 0, df[return_col] < 0, df[return_col] == 0]
    desc = ['Positive Rets', 'Negative Rets', 'Zero']
    posneg_summary = []
    for d, m in zip(desc, masks):
        vals = df.loc[m, return_col]
        count = vals.count()
        freq = 100 * count / total
        posneg_summary.append({
            'Desc': d,
            'Av Returns': vals.mean(),
            'Count': count,
            'Frequency': freq,
            'Freq. Adj Ret': vals.mean() * freq
        })

    # Std dev coverage
    std_summary = []
    for s in std_levels:
        lower, upper = mean_ret - s * std_ret, mean_ret + s * std_ret
        count = df[return_col].between(lower, upper).sum()
        std_summary.append({
            'StdDev': s,
            'LowerB': lower,
            'UpperB': upper,
            'Count': count,
            '% Count': 100 * count / total
        })

    return pd.DataFrame(posneg_summary), pd.DataFrame(std_summary), mean_ret, std_ret

def summarize_atr(df, return_col='True Range'):
    horizons = [5, 20, 60, 250, 750, 1250, 2500, 5000, 12500]
    labels = ['1 Week', '1 Month', '1 Quarter', '1 Year', '3 Years', '5 Years', '10 Years', '20 Years', '50 Years']
    return pd.DataFrame([
        {'Horizon': label, 'ATR': df[return_col].tail(days).mean() * 100}
        for label, days in zip(labels, horizons)
    ])

def plot_return_distribution(df, mean_ret, std_ret, return_col='C-C Return', bins=15):
    print(f'Columns: {df.columns}')
    bin_lims = np.linspace(mean_ret - 3 * std_ret, mean_ret + 3 * std_ret, bins)

    df = df.dropna(subset=[return_col])
    # Build distribution table
    DoR = pd.DataFrame(columns=['Bins', 'Frequency', 'Range', 'Probability', 'Cum. Prob.'])
    DoR['Bins'] = bin_lims
    freqs, labels = [], []

    for i, val in enumerate(bin_lims):
        if i == 0:
            count = (df[return_col] <= val).sum()
            label = f"Less than {val*100:.3f}%"
        elif i == len(bin_lims) - 1:
            count = (df[return_col] > val).sum()
            label = f"More than {val*100:.3f}%"
        else:
            lower = bin_lims[i - 1]
            count = ((df[return_col] > lower) & (df[return_col] <= val)).sum()
            label = f"{lower*100:.3f}% to {val*100:.3f}%"
        freqs.append(count)
        labels.append(label)

    DoR['Frequency'] = freqs
    DoR['Range'] = labels
    DoR['Probability'] = 100 * DoR['Frequency'] / len(df)
    DoR['Cum. Prob.'] = DoR['Probability'].cumsum()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5), tight_layout=True)
    ax.grid(visible=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.6)
    N, _, patches = ax.hist(df[return_col], bins=bin_lims)

    fracs = (N ** 0.2) / N.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    for frac, patch in zip(fracs, patches):
        patch.set_facecolor(plt.cm.viridis(norm(frac)))

    # Highlight latest return
    latest_val = df[return_col].iloc[-1]
    ax.plot(latest_val, 0, 'ro')  # Red dot
    rank = (df[return_col] < latest_val).sum() / len(df)
    txt = f"{df.index[-1]} {latest_val:.5f} | %rank: {rank:.2f}"
    ax.annotate(txt, (latest_val, 0), xytext=(latest_val, max(N)*0.1),
                arrowprops=dict(arrowstyle="->", facecolor='red'), color='red')
    

    ax.set_xticks(bin_lims)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {return_col}')
    plt.tight_layout()
    plt.show()
    
    print(df[return_col].describe())
    
    return DoR

def DoR_Analysis(df):
    tickers = df.columns.get_level_values(0).unique().tolist()
    def DoR_Analysis_helper(value1):
        df_individual = df[value1].copy()

        # Combined return analysis
        posneg_df, std_df, mean_ret, std_ret = analyze_returns(df_individual)
        
        # Plot distribution
        DoR = plot_return_distribution(df_individual, mean_ret, std_ret)
        
        # ATR Summary
        atr_df = summarize_atr(df_individual)
        
        # Display all
        display(std_df)
        display(posneg_df)
        display(DoR)
        display(atr_df)

    controls = widgets.interactive(DoR_Analysis_helper, value1=tickers)
    display(controls)


def plot_posneg_area(df, return_col='C-C Return'):
    desc = return_col
    df = df.dropna(subset=[desc]).copy()
    
    df.loc[:, 'pos'] = np.maximum(df[desc], 0)
    df.loc[:, 'neg'] = np.minimum(df[desc], 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['pos'], name='Positive', fill='tozeroy', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=df['neg'], name='Negative', fill='tozeroy', line=dict(color='red')))

    fig.update_layout(title=f"<b>Positive & Negative {desc}</b>", title_x=0.5,
                      margin=dict(l=20, r=30, t=50, b=30),
                      xaxis=dict(rangeslider_visible=True))
    fig.show()


def full_DoR_analysis(df, return_col):    
    print(f"\n{return_col} Distribution:\n")
    posneg_df, std_df, mean_ret, std_ret = analyze_returns(df, return_col=return_col)
    DoR = plot_return_distribution(df, mean_ret, std_ret, return_col=return_col)
    atr_df = summarize_atr(df, return_col=return_col)
    
    display(std_df)
    display(posneg_df)
    display(DoR)
    display(atr_df)
    
    # Plot positive/negative area chart
    plot_posneg_area(df[[return_col]], return_col)
