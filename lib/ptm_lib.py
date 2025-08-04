import ipywidgets as widgets
from IPython.display import display
from datetime import date, timedelta

import yfinance as yf
import pandas as pd
import numpy as np
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = 'notebook'
# pio.renderers.default = "jupyterlab"

import re
import requests

FRED_URL = 'https://api.stlouisfed.org/fred'
API_KEY = '4c65ec00c7f4c4f6c8c857294433952b'

def loadNupdate(tickers, from_date, to_date, freq, file):
    """
    Loads cached data from CSV or downloads it if missing, and updates it if out of date.
    """
    import os
    from datetime import timedelta

    path = os.path.join('datasets', f'{file}.csv')

    try:
        # Try loading local CSV
        data = pd.read_csv(path, index_col=0, dayfirst=True)
        data.index = pd.to_datetime(data.index, errors='coerce')
        if data.index.isnull().any():
            raise ValueError("Invalid date entries in CSV")
        data = data.reindex(columns=tickers)
    except (FileNotFoundError, ValueError):
        # If file doesn't exist or has bad dates, fetch fresh data
        data = load_data(tickers, from_date, to_date, freq)
        data.to_csv(path)
        return data

    # Mapping of frequency shorthand to timedelta estimates
    freq_key = freq[0].lower()
    offset = {'d': 1, 'w': 5, 'm': 30, 'q': 90, 'a': 260}
    next_timestamp = data.index[-1] + timedelta(days=offset.get(freq_key, 1))

    # Check if data needs updating
    if next_timestamp < pd.Timestamp(to_date):
        df = load_data(tickers, next_timestamp.date(), to_date, freq)
        if df.empty:
            print('No new data')
        else:
            df.index = pd.to_datetime(df.index)
            data = pd.concat([data, df])
            data = data[~data.index.duplicated(keep='last')].sort_index()
            data.to_csv(path)
            print('\nDataset updated and saved\n')

    return data

    

def load_data(tickers, from_date, to_date, freq):
    """
    Downloads price data using custom FRED-based loader or falls back to yfinance.
    """
    try:
        df = ticker_download(tickers, from_date, freq, verbose=False)
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("FRED download returned invalid or empty data.")
    except Exception as e:
        print(f"FRED download failed: {e}")
        df = fallback_yfinance(tickers, from_date, to_date, freq)

    df.index = pd.to_datetime(df.index, errors='coerce')
    if df.index.isnull().any():
        print("Warning: Dropping rows with invalid dates.")
        df = df[~df.index.isnull()]

    df.sort_index(inplace=True)
    df = df.reindex(columns=tickers)
    return df

def fallback_yfinance(tickers, from_date, to_date, freq):
    print("\nFalling back to yfinance...\n")
    interval = "1" + freq.lower()
    df = yf.download(tickers, start=from_date, end=to_date, interval=interval, auto_adjust=False)

    if isinstance(df.columns, pd.MultiIndex):
        df = df['Adj Close']
    else:
        df = df[['Adj Close']].rename(columns={'Adj Close': tickers[0]})

    return df.apply(pd.to_numeric, errors='coerce')




def ticker_download(tickers, from_date, freq, verbose=True):
    """
    Downloads time series data for given FRED tickers starting from `from_date` using the global API key.
    """
    df = pd.DataFrame()

    for ser in tickers:
        url = (
            f"{FRED_URL}/series/observations?"
            f"series_id={ser}&api_key={API_KEY}&file_type=json"
            f"&observation_start={from_date}&frequency={freq}"
        )
        try:
            response = requests.get(url)
            response.raise_for_status()
            tmp_json = response.json()

            if 'observations' not in tmp_json or not tmp_json['observations']:
                if verbose:
                    print(f"No observations found for {ser}")
                continue

            new_set = pd.DataFrame(tmp_json['observations'], columns=['date', 'value'])
            new_set.rename(columns={'value': ser}, inplace=True)
            new_set['date'] = pd.to_datetime(new_set['date'], errors='coerce')
            new_set.set_index('date', inplace=True)
            new_set[ser] = pd.to_numeric(new_set[ser], errors='coerce')
            new_set.dropna(inplace=True)

            df = pd.merge(df, new_set, how='outer', left_index=True, right_index=True)

        except Exception as e:
            if verbose:
                print(f"Error fetching {ser} from FRED: {e}")
            continue

    df.sort_index(inplace=True)
    df = df[~df.index.duplicated()]
    return df



################################## SPREAD ANALYSIS ####################################

def spread_analysis(data, tickers):
    def spread_analysis_helper(t1, t2):
        if t1 == t2:
            print("Please select two different tickers.")
            return

        # Extract 'Adj Close' for both tickers
        try:
            x1 = data[(t1, 'Adj Close')]
            x2 = data[(t2, 'Adj Close')]
        except:
            x1 = data[t1]
            x2 = data[t2]

        descriptions = get_ticker_descriptions([t1, t2])
        ticker1_desc = descriptions.get(t1, t1)
        ticker2_desc = descriptions.get(t2, t2)
        ticker1_desc = ticker1_desc[:70]
        ticker2_desc = ticker2_desc[:70]

    
        # Align on index and compute spread
        df = pd.DataFrame({
            'Spread': x1 - x2
        }).dropna()

        df['pos'] = np.maximum(df['Spread'], 0)
        df['neg'] = np.minimum(df['Spread'], 0)

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['pos'], name='Positive', fill='tozeroy'))
        fig.add_trace(go.Scatter(x=df.index, y=df['neg'], name='Negative', fill='tozeroy'))

        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=2, label="2y", step="year", stepmode="backward"),
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(count=10, label="10y", step="year", stepmode="backward"),
                    dict(step="all")
                ]
            )
        )

        fig.update_layout(
            title=dict(
                text=f"<b>{ticker1_desc} vs<br>{ticker2_desc} Spread</b>",
                x=0.5,
                font=dict(
                    size=12  # adjust size as needed
                )
            ),
            margin=dict(l=40, r=10, t=90, b=20)
        )

        fig.show()

    descriptions = get_ticker_descriptions(tickers)
    df = pd.DataFrame(list(descriptions.items()), columns=['Ticker', 'Description'])
    pd.set_option('display.max_colwidth', None)  # Show full descriptions
    display(df)

    controls = widgets.interactive(
        spread_analysis_helper,
        t1=widgets.Dropdown(options=tickers, description='Ticker 1'),
        t2=widgets.Dropdown(options=tickers, description='Ticker 2')
    )
    display(controls)


    
def returns(df, col, shift_periods):
    """ Function that takes as input a DataFrame (df), and calculates the returns of column (col) between n shifted_periods.                           
    """
    new_col = (df[col]/df[col].shift(shift_periods))-1 
    return new_col
    

    

def plot_select(data, tickers):
    def plot_ticker(tic):
        if tic not in data.columns:
            print(f"{tic} not found in data.")
            return
        descriptions = get_ticker_descriptions([tic])
        desc = descriptions.get(tic, tic)
        
        series = data[tic].dropna()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name=tic))

        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=2, label="2y", step="year", stepmode="backward"),
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(step="all")
                ]
            )
        )

        fig.update_layout(
            title=dict(
                text=f"<b>{desc} — Yield</b>",  # You can adjust to "Price" or anything else
                x=0.5,
                font=dict(size=16)
            ),
            margin=dict(l=20, r=20, t=90, b=20)
        )
        fig.show()

    dropdown = widgets.Dropdown(options=tickers, description='Ticker:')
    controls = widgets.interactive(plot_ticker, tic=dropdown)
    display(controls)






####################### INDEXES MARKET ANALYSIS ##########################

def market_analysis(data, tickers):    

    def market_analysis_helper(tic1):
        df = data[tic1].copy()
        df = df.dropna()
        df['RollHigh'] = df['High'].expanding().max()
        df['RollBearLevel'] = df['RollHigh'] * 0.8
        df['BullIndex'] = np.where(df['Adj Close'] > df['RollBearLevel'], df['Adj Close'], np.nan)
        df['BearIndex'] = np.where(df['Adj Close'] <= df['RollBearLevel'], df['Adj Close'], np.nan)
        descriptions = get_ticker_descriptions([tic1])
        desc = descriptions.get(tic1, tic1)
        
        
        fig = go.Figure()
        
        # Find the first non-NaN index safely
        first_valid_bull = df['BullIndex'].first_valid_index()
        first_valid_bear = df['BearIndex'].first_valid_index()
        
        if first_valid_bull is not None and first_valid_bear is not None:
            first_non_nan_index = min(first_valid_bull, first_valid_bear)
        elif first_valid_bull is not None:
            first_non_nan_index = first_valid_bull
        elif first_valid_bear is not None:
            first_non_nan_index = first_valid_bear
        else:
            first_non_nan_index = None  # No valid data

        if first_non_nan_index is not None:
            fig.add_trace(go.Scatter(x=df.loc[first_non_nan_index:].index, 
                                     y=df.loc[first_non_nan_index:, 'BullIndex'], 
                                     mode="lines", line=dict(color='green', width=1.5)))
            fig.add_trace(go.Scatter(x=df.loc[first_non_nan_index:].index, 
                                     y=df.loc[first_non_nan_index:, 'BearIndex'], 
                                     mode="lines", line=dict(color='red', width=1.5)))
            fig.update_yaxes(title_text=tic1)
            
            # Annotation fix: Ensure indx_lim is valid before using it
            ax_lef, ax_rig, ay_lef, ay_rig = -70, 30, -30, 30

            if pd.notna(df['BullIndex'].iloc[-1]):
                indx_lim = df['BearIndex'].last_valid_index()
                if indx_lim is not None and not df.loc[indx_lim:, 'BullIndex'].dropna().empty:
                    indx = df.loc[indx_lim:, 'BullIndex'].idxmax()
                    ypos = df.loc[indx, 'BullIndex']
                    txt = indx.strftime('%d-%b-%y') + " " + "{:.2f}".format(ypos)
                    fig.add_annotation(x=indx, y=ypos, ax=ax_lef, ay=ay_lef, text=txt, showarrow=True, 
                                       arrowhead=1, arrowcolor='green', font=dict(color='green', size=14))

                indx = df.index[-1]
                if pd.notna(df.loc[indx, 'BullIndex']):
                    ypos = df.loc[indx, 'BullIndex']
                    txt = "{:.2f}".format(ypos)
                    fig.add_annotation(x=indx, y=ypos, ax=ax_rig, ay=ay_rig, text=txt, showarrow=True, 
                                       arrowhead=1, arrowcolor='green', font=dict(color='green', size=14))
            else:
                indx_lim = df['BullIndex'].last_valid_index()
                if indx_lim is not None and not df.loc[indx_lim:, 'BearIndex'].dropna().empty:
                    indx = df.loc[indx_lim:, 'BearIndex'].idxmin()
                    ypos = df.loc[indx, 'BearIndex']
                    txt = indx.strftime('%d-%b-%y') + " " + "{:.2f}".format(ypos)
                    fig.add_annotation(x=indx, y=ypos, ax=ax_lef, ay=ay_lef, text=txt, showarrow=True, 
                                       arrowhead=1, arrowcolor='red', font=dict(color='red', size=14))

                indx = df.index[-1]
                if pd.notna(df.loc[indx, 'BearIndex']):
                    ypos = df.loc[indx, 'BearIndex']
                    txt = "{:.2f}".format(ypos)
                    fig.add_annotation(x=indx, y=ypos, ax=ax_rig, ay=ay_rig, text=txt, showarrow=True, 
                                       arrowhead=1, arrowcolor='red', font=dict(color='red', size=14))

            fig.update_xaxes(rangeslider_visible=True)
            fig.update_layout(title_text=f"{desc} - Bull/Bear Zone", title_x=0.5)
            fig.update_layout(margin=dict(l=50, r=30, t=60, b=20))
            fig.update_traces(showlegend=False)
            fig.show()

        fig = go.Figure()    
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Date'] + list(df.columns),
                    font=dict(size=10),
                    align="center"
                ),
                cells=dict(
                    values=[df.index.strftime('%d-%m-%Y').tolist()] + [df[k].round(3).tolist() for k in df.columns],
                    align="left"
                )
            )
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        fig.show()

    controls = widgets.interactive(market_analysis_helper, tic1=widgets.Dropdown(options=tickers))
    display(controls)


def load_market_index_data(from_date, today, file, tickers):
    try:
        data = pd.read_csv(f'datasets/{file}.csv', index_col=0, parse_dates=True, dayfirst=True, header=[0, 1])
    except Exception:
        data = yf.download(tickers, from_date, today, interval="1d", auto_adjust=False)
        data = data.swaplevel(axis=1)
        data = data.loc[:, pd.IndexSlice[:, ['Adj Close', 'High']]]
        data = data.astype('float64')
        data.index = data.index.strftime('%d-%m-%Y')
        data = data.loc[:, tickers]
        data.to_csv(f'datasets/{file}.csv', index=True)

    return data.loc[:, tickers]



################################# GDP ANALYSIS  ################################# 
    
def merge_yoy_ticker(data, ticker, from_date, to_date, interval, ref_col_index, window):
    """
    Merges a Yahoo Finance ticker's adjusted close prices into a macroeconomic dataset,
    aligns them by date, and computes Year-over-Year (YoY) returns for all columns.

    Parameters:
        data (pd.DataFrame): Existing macroeconomic data with datetime index
        ticker (str): Yahoo Finance ticker symbol (e.g., '^GSPC')
        from_date (datetime.date): Start date for ticker data
        to_date (datetime.date): End date for ticker data
        interval (str): Data interval (e.g., '1mo', '3mo')
        ref_col_index (int): Index of the macro column to compare against
        window (int): Rolling correlation window size (used later)

    Returns:
        pd.DataFrame: Combined dataset with YoY columns added
    """
    periods_to_year = {'d': 260, 'w': 52, 'm': 12, 'q': 3, '1d': 260, '1wk': 52, '1mo': 12, '3mo': 4}
    if interval not in periods_to_year:
        raise ValueError(f"Invalid interval: {interval}. Allowed: {list(periods_to_year.keys())}")

    df = yf.download(ticker, start=from_date, end=to_date, interval=interval, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns.values]

    adj_close_col = next((col for col in df.columns if 'Adj Close' in col), None)
    if adj_close_col is None:
        raise KeyError(f"'Adj Close' column not found in downloaded data for {ticker}.")

    df = df[[adj_close_col]].copy()
    df.rename(columns={adj_close_col: ticker}, inplace=True)

    # Merge and compute YoY
    data = pd.merge(data, df, how='outer', left_index=True, right_index=True)

    for col in data.columns:
        data[col + ' YoY'] = returns(data, col, periods_to_year[interval])

    return data.dropna()



import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_yoy_lagged_correlation(data, window, ticker):
    """
    Plots rolling correlation between a macroeconomic YoY series and a financial ticker's YoY values
    at multiple lags. Highlights the lag with the highest average correlation.

    Parameters:
        data (pd.DataFrame): DataFrame with YoY columns for macro and ticker
        window (int): Rolling correlation window size (in periods)
        ticker (str): Ticker column name in the DataFrame (e.g., '^GSPC')

    Returns:
        dict: Average correlation values by lag
    """
    macro_col = data.columns[1]
    macro_yoy = macro_col + ' YoY'
    ticker_yoy = ticker + ' YoY'

    # Get descriptions
    descriptions = get_ticker_descriptions([ticker, macro_col])
    ticker_desc = descriptions.get(ticker, ticker)
    macro_desc = descriptions.get(macro_col, macro_col)

    lags = {
        'No lag': 0,
        '3-month lag': 1,
        '6-month lag': 2,
        '9-month lag': 3,
        '12-month lag': 4
    }

    print(f"\nRolling {window}-period correlation between {ticker_desc} and {macro_desc}")
    avg_corrs = {}

    fig_corr = go.Figure()
    for label, shift in lags.items():
        shifted_corr = data[macro_yoy].rolling(window).corr(data[ticker_yoy].shift(shift)).dropna()
        avg_corr = shifted_corr.mean()
        avg_corrs[label] = avg_corr
        print(f"{label:<15}: {avg_corr:.4f}")
        line_style = dict(width=2)
        if label == 'No lag':
            line_style.update(color='black', dash='dash')
        fig_corr.add_trace(go.Scatter(
            x=shifted_corr.index,
            y=shifted_corr,
            mode='lines',
            name=label,
            line=line_style
        ))

    # Highlight best lag
    best_lag = max(avg_corrs, key=avg_corrs.get)
    fig_corr.add_annotation(
        text=f"Best lag: {best_lag} ({avg_corrs[best_lag]:.2f})",
        xref="paper", yref="paper",
        x=0.01, y=1.1,
        showarrow=False,
        font=dict(size=13)
    )

    fig_corr.update_layout(
        title=f"Rolling Correlation: {ticker_desc} vs {macro_desc}",
        xaxis_title="Date",
        yaxis_title="Correlation",
        title_x=0.5,
        height=500
    )
    fig_corr.show()

    # Overlay time series
    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ts.add_trace(go.Scatter(x=data.index, y=data[macro_yoy], name=macro_desc, line=dict(color='blue')), secondary_y=False)
    fig_ts.add_trace(go.Scatter(x=data.index, y=data[ticker_yoy], name=ticker_desc, line=dict(color='orange')), secondary_y=True)

    fig_ts.update_layout(
        title=f"YoY Comparison: {ticker_desc} vs {macro_desc}",
        xaxis_title="Date",
        title_x=0.5,
        height=500
    )
    fig_ts.update_yaxes(title_text=macro_desc, secondary_y=False)
    fig_ts.update_yaxes(title_text=ticker_desc, secondary_y=True)
    fig_ts.show()

    print(f"\nConclusion: The highest rolling correlation occurs at {best_lag},")
    print(f"suggesting that {ticker_desc} leads {macro_desc} by approximately {lags[best_lag] * 3} months.")

    return avg_corrs





def get_ticker_descriptions(tickers):
    descriptions = {}
    
    for ticker in tickers:
        description = None

        # 1. Try yfinance (shortName preferred)
        try:
            info = yf.Ticker(ticker).info
            name = info.get("shortName") or info.get("longName")
            if isinstance(name, str):
                description = re.sub(r'\s+', ' ', name).strip()
        except:
            pass

        # 2. Try FRED
        if description is None:
            fred_id = ticker.replace('^', '')
            try:
                url = f"{FRED_URL}/series"
                params = {
                    'series_id': fred_id,
                    'api_key': API_KEY,
                    'file_type': 'json'
                }
                resp = requests.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    if "seriess" in data and data["seriess"]:
                        title = data["seriess"][0].get("title")
                        if isinstance(title, str):
                            description = re.sub(r'\s+', ' ', title).strip()
            except:
                pass

        # 3. Fallback
        if description is None:
            description = ticker

        descriptions[ticker] = description

    return descriptions




############################# YIELD CURVE ################################

import ipywidgets as widgets
import plotly.graph_objs as go
from IPython.display import display

def yield_curve(dataset, tips):
    date_slider1 = widgets.SelectionSlider(
        options=dataset.index,
        description='Blue Select:',
        continuous_update=False,
        layout={'width': '600px'}
    )

    date_slider2 = widgets.SelectionSlider(
        options=dataset.index,
        description='Orange Select:',
        continuous_update=False,
        layout={'width': '600px'}
    )

    def plot_curve(day1, day2, tips):
        if tips == 'TIPS':
            labels = ['FedFund','1mo','3mo','6mo','1yr','2yr','3yr','5yr','7yr','10yr','20yr']
            y1 = dataset.loc[day1].iloc[:len(labels)]
            y2 = dataset.loc[day2].iloc[:len(labels)]
            color1 = 'rgba(31, 119, 180, 0.6)'
            color2 = 'rgba(255, 127, 14, 0.6)'
        else:
            labels = ['5yr-TIPS','7yr-TIPS','10yr-TIPS','20yr-TIPS','30yr-TIPS']
            y1 = dataset.loc[day1].iloc[-len(labels):]
            y2 = dataset.loc[day2].iloc[-len(labels):]
            color1 = 'rgba(31, 119, 180, 0.6)'
            color2 = 'rgba(255, 127, 14, 0.6)'

        def make_trace(y, label, name):
            return go.Scatter(
                x=labels, y=y,
                mode='lines+markers+text',
                name=name,
                marker=dict(size=10, line=dict(color='black', width=1))
            )

        def make_annotations(x_vals, y_vals, color, x_shift):
            return [dict(
                x=x, y=y,
                text=str(y),
                showarrow=False,
                font=dict(family="Arial", size=12, color="white"),
                bgcolor=color,
                bordercolor=color.replace('0.6', '0.8'),
                borderwidth=1,
                borderpad=4,
                xshift=x_shift,
                yshift=17,
                xref='x', yref='y'
            ) for x, y in zip(x_vals, y_vals)]

        fig = go.Figure(
            data=[make_trace(y1, labels, str(day1.date())), make_trace(y2, labels, str(day2.date()))],
            layout=go.Layout(
                title=dict(text='<b>US Yield Curve - Comparative Analysis</b>', x=0.5),
                yaxis=dict(title='<b>Percent</b>'),
                legend=dict(orientation='v', x=-0.15, y=-0.3),
                annotations=make_annotations(labels, y1, color1, -17) + make_annotations(labels, y2, color2, 17)
            )
        )

        fig.show()

    controls = widgets.interactive(plot_curve, day1=date_slider1, day2=date_slider2, tips=tips)
    display(controls)


    