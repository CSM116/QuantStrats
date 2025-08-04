
"""
visuals.py

Contains plotting functions for visualizing signals, price action, and performance metrics.
Functions accept pre-computed Series or DataFrames and use Plotly for interactive charts.

Does not compute signals or analytics — expects clean, formatted inputs.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_price_with_signals(price, long_entries=None, short_entries=None, title="Price with Signals"):
    """
    Plots price series with long and short signal markers.
    Expects:
    - `price`: Series of price
    - `long_entries`: Index values or Series where longs occur
    - `short_entries`: Index values or Series where shorts occur
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price.index, y=price, name='Price'))

    if long_entries is not None:
        fig.add_trace(go.Scatter(
            x=long_entries.index if isinstance(long_entries, pd.Series) else long_entries,
            y=price.loc[long_entries],
            mode='markers',
            name='Long',
            marker=dict(color='green', size=8, symbol='triangle-up')
        ))

    if short_entries is not None:
        fig.add_trace(go.Scatter(
            x=short_entries.index if isinstance(short_entries, pd.Series) else short_entries,
            y=price.loc[short_entries],
            mode='markers',
            name='Short',
            marker=dict(color='red', size=8, symbol='triangle-down')
        ))
        
    go_plot_helper(fig, title)
    fig.show()



import pandas as pd
import plotly.graph_objects as go

def plot_series(series, title="Time Series", y_label=None, trace_name=None):
    """
    Plot one or more time series using Plotly.

    Parameters:
    - series: pd.Series or pd.DataFrame – time series to plot (indexed by datetime)
    - title: str – plot title
    - y_label: str – label for the y-axis (optional)
    - trace_name: str or list of str – name(s) for the line trace(s) (optional)
    """
    fig = go.Figure()

    if isinstance(series, pd.Series):
        name = trace_name if isinstance(trace_name, str) else trace_name[0] if trace_name else series.name or title
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name=name))
    elif isinstance(series, pd.DataFrame):
        cols = series.columns
        if isinstance(trace_name, list) and len(trace_name) == len(cols):
            names = trace_name
        else:
            names = cols
        for col, name in zip(cols, names):
            fig.add_trace(go.Scatter(x=series.index, y=series[col], name=name))
    else:
        raise ValueError("Input must be a pandas Series or DataFrame")

    fig.update_layout(
        title=title,
        yaxis_title=y_label or "",
        xaxis_title="Date",
        showlegend=True
    )
    go_plot_helper(fig, title)
    fig.show()




################### STYLE HELPER ##################

def go_plot_helper(fig, title):
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
            text=title, 
            x=0.5,
            font=dict(size=12)
        ),
        margin=dict(l=40, r=10, t=90, b=20)
    )