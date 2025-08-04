import wrds
import pandas as pd

import numpy as np
import os



def download_sp500_membership(output_dir, redownload, db):
    sp500_file = f'{output_dir}/sp500_constituents.parquet'
    if redownload:
        sp500_mem = db.raw_sql("""
            SELECT 
                permno,        
                mbrstartdt, 
                mbrenddt
            FROM crsp.dsp500list_v2
        """, date_cols=['mbrstartdt', 'mbrenddt'])
        sp500_mem.to_parquet(sp500_file, index=False, compression='brotli')
    else:
        sp500_mem = pd.read_parquet(sp500_file, engine='pyarrow')
        sp500_mem['mbrstartdt'] = pd.to_datetime(sp500_mem['mbrstartdt'])
        sp500_mem['mbrenddt'] = pd.to_datetime(sp500_mem['mbrenddt'])
    return sp500_mem



def download_crsp_daily_data(db, start_date, end_date, output_dir, redownload, source):
    assert source in ['wrds', 'raw'], "source must be 'wrds' or 'raw'"
    table = 'crsp.wrds_dsfv2_query' if source == 'wrds' else 'crsp.dsf_v2'
    output_path = f'{output_dir}/{start_date}_{end_date}_{source}_dsfv2_eqty.parquet'


    if not redownload and os.path.exists(output_path):
        price_df = pd.read_parquet(output_path, engine='pyarrow')
        price_df['dlycaldt'] = pd.to_datetime(price_df['dlycaldt'])
        if source == 'wrds' and 'dispaydt' in price_df.columns:
            price_df['dispaydt'] = pd.to_datetime(price_df['dispaydt'])
        print(f"Loaded Daily data {output_path}")    
    else:
        query = f"""
            SELECT
                permno,
                primaryexch,
                tradingstatusflg,
                securitytype,
                ticker,
                permco,
                siccd,
                {'naics, icbindustry, disdivamt, dispaydt' if source == 'wrds' else ''}
                dlycaldt,
                dlyprc,
                dlycap,
                dlyret,
                dlyretx,
                dlyvol,
                dlyclose,
                dlylow,
                dlyhigh,
                dlyopen,
                dlynumtrd,
                shrout,
                dlycumfacpr,
                (dlyprc / NULLIF(dlycumfacpr, 0)) AS adj_prc
            FROM {table}
            WHERE dlycaldt BETWEEN '{start_date}' AND '{end_date}'
              AND securitytype = 'EQTY'
        """
        date_cols = ['dlycaldt', 'dispaydt'] if source == 'wrds' else ['dlycaldt']
        price_df = db.raw_sql(query, date_cols=date_cols)
        price_df.to_parquet(output_path, index=False, compression='brotli')
        print(f"Saved Daily data to {output_path}")
        
    return price_df



def merge_sp500_with_crsp(price_df, sp500_df, output_dir, source, merge_flag):
    merged_file = f'{output_dir}/{source}_sp500_merged.parquet'
    if merge_flag:
        merged = pd.merge(price_df, sp500_df, on='permno')
        merged = merged[
            (merged['dlycaldt'] >= merged['mbrstartdt']) &
            (merged['dlycaldt'] <= merged['mbrenddt'])
        ].copy()
        merged.drop(columns=['mbrstartdt', 'mbrenddt'], inplace=True)
        merged.to_parquet(merged_file, index=False, compression='brotli')
    else:
        merged = pd.read_parquet(merged_file, engine='pyarrow')
    return merged



def main_download_sp500_data(db, start_date='2000-01-01',
                             end_date='2024-12-31',
                             output_dir='datasets',
                             redownloadSP=False,
                             redownloadDSF=False,
                             source='wrds'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    merge_flag = redownloadSP or redownloadDSF
    if merge_flag:
        sp500_df = download_sp500_membership(output_dir, redownloadSP, db)
        price_df = download_crsp_daily_data(start_date, end_date, output_dir, redownloadDSF, source, db)
        merged_df = merge_sp500_with_crsp(price_df, sp500_df, output_dir, source, merge_flag)
    else:
        merged_df = merge_sp500_with_crsp(None, None, output_dir, source, merge_flag)    

    return merged_df



def get_sp500_daily_fullset(db, start_date='2000-01-01', output_dir='datasets'):
    db = get_wrds_connection()

    query = f"""
        SELECT *
        FROM crsp.dsp500list_v2 AS a
        JOIN crsp.dsf_v2 AS b
          ON a.permno = b.permno
        WHERE b.dlycaldt >= a.mbrstartdt
          AND b.dlycaldt <= a.mbrenddt
          AND b.dlycaldt >= '{start_date}'
        ORDER BY b.dlycaldt
    """

    df = db.raw_sql(query, date_cols=['dlycaldt', 'mbrstartdt', 'mbrenddt'])

    # Optional: save to disk
    file_name = f'{output_dir}/sp500_daily_returns.parquet'
    df.to_parquet(file_name, index=False, compression='brotli')

    return df




def compute_advanced_metrics(df, risk_free=0.075, equity_risk_premium=0.09, beta=1.0):
    df = df.copy()
    df = df.sort_values(['gvkey', 'datadate'])
    
    # ROIC
    df['roic'] = df['nopat'] / df['icaptq']
    # Cash ROIC (FCF / invested capital)
    df['croic'] = df['fcf'] / df['icaptq']

    # Lagged FCF for growth
    df['fcf_lag'] = df.groupby('gvkey')['fcf'].shift(1)
    df['fcf_growth'] = df['fcf'] / df['fcf_lag']

    # FCF to EBIT
    df['fcf_to_ebit'] = df['fcf'] / df['oiadpq']

    # EBITDA - Capex to IC
    df['ebitda_minus_capex_to_ic'] = (df['ebitda'] - df['capx_q']) / df['icaptq']

    # # Cost of capital (WACC)
    # df['cost_debt'] = (df['xintq'] / (df['tot_debt'])).replace([np.inf, -np.inf], np.nan) * (1 - (df['txtq'] / df['piq']))
    # df['cost_equity'] = risk_free + beta * equity_risk_premium    
    # df['w_debt'] = (df['tot_debt']) / df['tot_capt']
    # df['w_equity'] = df['ceqq'] / df['tot_capt']
    # df['wacc'] = df['cost_debt'] * df['w_debt'] + df['cost_equity'] * df['w_equity']

    # # Economic Profit
    # df['ec_profit'] = (df['roic'] - df['wacc']) * df['icaptq']

    # EV/EBITDA
    df['ev_to_ebitda'] = df['ev'] / df['ebitda']

    return df



def download_fundamentals(db, start_date, end_date, output_dir='datasets', redownload=False):
    output_path = f'{output_dir}/{start_date}_{end_date}_fundq.parquet'
    
    if not redownload and os.path.exists(output_path):
        fundq = pd.read_parquet(output_path)
        print(f"Loaded {output_path}")
    else:    
        fundq = db.raw_sql(f"""
            SELECT
                f.gvkey,        -- Compustat unique company ID
                f.tic,          -- Ticker symbol
                f.conm,         -- Company name
                f.datadate,     -- Fiscal period end date (quarter end)
                f.pdateq,       -- Date Compustat published the data (use for bias-free timing)
                f.fyearq,       -- Fiscal year (of the quarter)
                f.fqtr,         -- Fiscal quarter number (1 to 4)
                f.rdq,          -- Report date of earnings (actual announcement)
                f.costat,       -- Company status ('A' = active, 'I' = inactive)
                f.curcdq,       -- Currency code of the report (usually 'USD')
                f.datafmt,      -- Data format (should be 'STD' for standardized)
                f.indfmt,       -- Industry format ('INDL' for industrial format)
                f.consol,       -- Consolidation code ('C' = consolidated)
                f.mkvaltq,      -- Market value (common shares out * month end price)
                f.prccq,        -- Closing price
                
                -- Balance sheet & financials
                f.atq,          -- Total assets (quarterly)
                f.ceqq,         -- Common equity (quarterly)
                f.cheq,         -- Cash and equivalents
                f.chechy,       -- Cash and Cash equivalents
                f.cshopq,       -- Shares repurchased (quarterly)
                f.cshoq,        -- Shares outstanding (quarterly)
                f.dlcq,         -- Current debt (short-term borrowings)
                f.dlttq,        -- Long-term debt
                f.dpq,          -- Depreciation and amortization
                f.epspxq,       -- Basic EPS excluding extraordinary items
                f.epsx12,       -- Trailing 12-month EPS
                f.ibq,          -- Income before extraordinary items (quarterly)
                f.invtq,        -- Inventory
                f.ltq,          -- Total liabilities
                f.seqq,         -- Sharehlders Equity
                f.mibq,         -- Minority interest (quarterly)
                f.niq,          -- Net income
                f.ppentq,       -- Net property, plant, and equipment
                f.pstkq,        -- Preferred stock (quarterly)
                f.rectq,        -- Accounts receivable (quarterly)
                f.xintq,        -- Interest expense (quarterly)
                f.capxy,        -- Capital expenditures (ytd)
                f.oancfy,       -- Operating cash flow (ytd)
                f.icaptq,       -- Invested Capital total (quarterly)
                f.oiadpq,       -- Operating Income After Depreciation (EBIT)
                f.txtq,         -- Total income taxes
                f.piq,          -- Pretax income
            
                -- Industry classification from company table
                id.ggroup,      -- GICS group
                id.gind,        -- GICS industry
                id.gsector,     -- GICS sector
                id.sic,         -- Standard Industrial Classification code
        
                -- Simple Derived Metrics
                (f.mkvaltq + f.dlcq + f.dlttq - f.chechy) AS ev,                     -- Enterprise Value
                (f.niq / NULLIF(f.seqq, 0)) AS roe,                                  -- Return on Equity
                (f.niq / NULLIF(f.atq, 0)) AS roa,                                   -- Return on Assets
                (f.prccq / NULLIF(f.epspxq, 0)) AS pe_ratio,                         -- P/E ratio
                (f.ibq + f.dpq) AS ebitda,                                           -- Ebitda
                (f.dlcq + f.dlttq) AS tot_debt,                                      -- Total debt
                (f.dlcq + f.dlttq + f.seqq) AS tot_capt,                             -- Total capital
                (f.oiadpq * (f.txtq / NULLIF(f.piq, 0))) AS nopat,                   -- Net Operating Profit After Tax
                (f.dlttq + f.dlcq) / NULLIF(f.atq, 0) AS debt_to_assets,             -- Debt to assets
                f.cheq / NULLIF(f.atq, 0) AS cash_to_assets,                         -- Cash ratio
                (f.invtq + f.rectq) / NULLIF(f.atq, 0) AS wc_assets_to_assets,       -- Working capital assets to total assets

                -- YoY EPS growth 
                CASE
                    WHEN LAG(f.epsx12) OVER (PARTITION BY f.gvkey ORDER BY f.datadate) IS NOT NULL
                         AND f.epsx12 IS NOT NULL
                         AND LAG(f.epsx12) OVER (PARTITION BY f.gvkey ORDER BY f.datadate) != 0
                    THEN (
                        f.epsx12 - LAG(f.epsx12) OVER (PARTITION BY f.gvkey ORDER BY f.datadate)
                    ) / NULLIF(LAG(f.epsx12) OVER (PARTITION BY f.gvkey ORDER BY f.datadate), 0)
                    ELSE NULL
                END AS eps_yoy_growth,


                -- EPS Q-Q growth
                CASE
                    WHEN LAG(f.epspxq) OVER (PARTITION BY f.gvkey ORDER BY f.datadate) IS NOT NULL
                         AND f.epspxq IS NOT NULL
                         AND LAG(f.epspxq) OVER (PARTITION BY f.gvkey ORDER BY f.datadate) != 0
                    THEN (f.epspxq - LAG(f.epspxq) OVER (PARTITION BY f.gvkey ORDER BY f.datadate))
                         / NULLIF(LAG(f.epspxq) OVER (PARTITION BY f.gvkey ORDER BY f.datadate), 0)
                    ELSE NULL
                END AS eps_qoq_growth,

                -- Sales QoQ growth
                CASE
                    WHEN LAG(f.saleq) OVER (PARTITION BY f.gvkey ORDER BY f.datadate) IS NOT NULL
                         AND f.saleq IS NOT NULL
                         AND LAG(f.saleq) OVER (PARTITION BY f.gvkey ORDER BY f.datadate) != 0
                    THEN (
                        f.saleq - LAG(f.saleq) OVER (PARTITION BY f.gvkey ORDER BY f.datadate)
                    ) / NULLIF(LAG(f.saleq) OVER (PARTITION BY f.gvkey ORDER BY f.datadate), 0)
                    ELSE NULL
                END AS sales_qoq_growth,

                -- Sales YoY growth
                CASE
                    WHEN LAG(f.saleq, 4) OVER (PARTITION BY f.gvkey ORDER BY f.datadate) IS NOT NULL
                         AND f.saleq IS NOT NULL
                         AND LAG(f.saleq, 4) OVER (PARTITION BY f.gvkey ORDER BY f.datadate) != 0
                    THEN (
                        f.saleq - LAG(f.saleq, 4) OVER (PARTITION BY f.gvkey ORDER BY f.datadate)
                    ) / NULLIF(LAG(f.saleq, 4) OVER (PARTITION BY f.gvkey ORDER BY f.datadate), 0)
                    ELSE NULL
                END AS sales_yoy_growth,
                
                -- CapEx Quarterly
                CASE
                    WHEN f.fyearq = LAG(f.fyearq) OVER (PARTITION BY f.gvkey ORDER BY f.datadate)
                         AND f.capxy  != LAG(f.capxy)  OVER (PARTITION BY f.gvkey ORDER BY f.datadate)
                    THEN (
                        f.capxy - LAG(f.capxy) OVER (PARTITION BY f.gvkey ORDER BY f.datadate)
                    ) 
                    ELSE NULL
                END AS capx_q,
                
                -- Cash ROE = Δ Operating Cash Flow / Invested Capital
                CASE
                    WHEN f.fyearq = LAG(f.fyearq) OVER (PARTITION BY f.gvkey ORDER BY f.datadate)
                         AND f.oancfy != LAG(f.oancfy) OVER (PARTITION BY f.gvkey ORDER BY f.datadate)
                    THEN (f.oancfy - LAG(f.oancfy) OVER (PARTITION BY f.gvkey ORDER BY f.datadate)) 
                         / NULLIF(f.icaptq, 0)
                    ELSE NULL
                END AS croe,
                
                -- Free Cash Flow = Δ Operating CF - Δ CapEx
                CASE
                    WHEN f.fyearq = LAG(f.fyearq) OVER (PARTITION BY f.gvkey ORDER BY f.datadate)
                         AND f.oancfy != LAG(f.oancfy) OVER (PARTITION BY f.gvkey ORDER BY f.datadate)
                         AND f.capxy  != LAG(f.capxy)  OVER (PARTITION BY f.gvkey ORDER BY f.datadate)
                    THEN (f.oancfy - LAG(f.oancfy) OVER (PARTITION BY f.gvkey ORDER BY f.datadate)) 
                         - (f.capxy  - LAG(f.capxy)  OVER (PARTITION BY f.gvkey ORDER BY f.datadate))
                    ELSE NULL
                END AS fcf,
                
                -- FCF to EBIT = FCF / EBIT
                CASE
                    WHEN f.fyearq = LAG(f.fyearq) OVER (PARTITION BY f.gvkey ORDER BY f.datadate)
                         AND f.oancfy != LAG(f.oancfy) OVER (PARTITION BY f.gvkey ORDER BY f.datadate)
                         AND f.capxy  != LAG(f.capxy)  OVER (PARTITION BY f.gvkey ORDER BY f.datadate)
                    THEN (
                        (f.oancfy - LAG(f.oancfy) OVER (PARTITION BY f.gvkey ORDER BY f.datadate)) -
                        (f.capxy  - LAG(f.capxy)  OVER (PARTITION BY f.gvkey ORDER BY f.datadate))
                    ) / NULLIF(f.oiadpq, 0)
                    ELSE NULL
                END AS fcf_to_ebit,
                
                -- CapEx Coverage = FCF / Δ CapEx
                CASE
                    WHEN f.fyearq = LAG(f.fyearq) OVER (PARTITION BY f.gvkey ORDER BY f.datadate)
                         AND f.oancfy != LAG(f.oancfy) OVER (PARTITION BY f.gvkey ORDER BY f.datadate)
                         AND f.capxy  != LAG(f.capxy)  OVER (PARTITION BY f.gvkey ORDER BY f.datadate)
                         AND (f.capxy - LAG(f.capxy) OVER (PARTITION BY f.gvkey ORDER BY f.datadate)) != 0
                    THEN (
                        (f.oancfy - LAG(f.oancfy) OVER (PARTITION BY f.gvkey ORDER BY f.datadate)) -
                        (f.capxy  - LAG(f.capxy)  OVER (PARTITION BY f.gvkey ORDER BY f.datadate))
                    ) / NULLIF(
                        (f.capxy - LAG(f.capxy) OVER (PARTITION BY f.gvkey ORDER BY f.datadate)), 0
                    )
                    ELSE NULL
                END AS capex_coverage_ratio,
                
                -- CapEx to PP&E = Δ CapEx / PPE
                CASE
                    WHEN f.fyearq = LAG(f.fyearq) OVER (PARTITION BY f.gvkey ORDER BY f.datadate)
                         AND f.capxy  != LAG(f.capxy)  OVER (PARTITION BY f.gvkey ORDER BY f.datadate)
                         AND f.ppentq IS NOT NULL
                    THEN (
                        f.capxy - LAG(f.capxy) OVER (PARTITION BY f.gvkey ORDER BY f.datadate)
                    ) / NULLIF(f.ppentq, 0)
                    ELSE NULL
                END AS capex_to_ppe

                
            FROM comp_na_daily_all.fundq AS f
            LEFT JOIN (
                SELECT gvkey, ggroup, gind, gsector, sic
                FROM comp_na_daily_all.company
            ) AS id
            ON f.gvkey = id.gvkey
        
            WHERE f.pdateq BETWEEN '{start_date}' AND '{end_date}' -- use pdateq to simulate true availability
        
            AND f.consol = 'C'
            AND f.indfmt = 'INDL'
            AND f.datafmt = 'STD'
            AND f.curcdq = 'USD'
            AND f.costat IN ('A','I')
            AND f.datafqtr IS NOT NULL

        """)

        fundq = compute_advanced_metrics(fundq)
        fundq.to_parquet(output_path, index=False, compression='brotli')
        print(f"Saved Fundamentals to {output_path}")
        
    return fundq



def download_and_save_gvkey_permno_mapping(db, output_dir, redownload=False):
    """
    Download and save gvkey–permno mapping using crsp.ccmxpf_linktable.
    Saves as Parquet file.
    """
    output_path = f'{output_dir}/linktable.parquet'
    if not redownload and os.path.exists(output_path):
        mapping_df = pd.read_parquet(output_path)
        print(f"Loaded mapping from {output_path}")
    else:
        db = get_wrds_connection()
        query = """
            SELECT DISTINCT gvkey, lpermno AS permno, linkdt, linkenddt
            FROM crsp.ccmxpf_linktable
            WHERE linktype IN ('LU', 'LC') AND usedflag = 1  -- Standard CRSP equity links
              AND lpermno IS NOT NULL
              AND gvkey IS NOT NULL
        """
        mapping_df = db.raw_sql(query)
        mapping_df['linkdt'] = pd.to_datetime(mapping_df['linkdt'])
        mapping_df['linkenddt'] = pd.to_datetime(mapping_df['linkenddt'])
        mapping_df['permno'] = mapping_df['permno'].astype('Int64')  # pandas nullable int
        mapping_df.to_parquet(output_path, index=False, compression='brotli')
        print(f"Saved mapping to {output_path}")

    return mapping_df




import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def merge_single_permno(args):
    permno, df_daily, df_fundq, max_days_back = args

    merged = pd.merge_ordered(
        df_daily,
        df_fundq,
        left_on='dlycaldt',
        right_on='rdq',
        fill_method='ffill',
        suffixes=('', '_fundq')
    )
    merged = merged.dropna(subset=['rdq'])

    if max_days_back:
        merged = merged[merged['dlycaldt'] - merged['rdq'] <= pd.Timedelta(days=max_days_back)]

    return merged

    
def merge_fundq_and_daily_prices(
    fundq_df,
    daily_df,
    gvkey_permno_map,
    fundq_cols=None,
    daily_cols=None,
    max_days_back=None,
    n_jobs=4
):
    """
    Safe to run in JupyterLab. Parallel merge of daily prices and fundamentals by permno using merge_ordered.
    """

    fundq_df = fundq_df.copy().reset_index(drop=True)
    daily_df = daily_df.copy().reset_index(drop=True)
    mapping_df = gvkey_permno_map.copy().reset_index(drop=True)

    fundq_df['gvkey'] = fundq_df['gvkey'].astype(str).str.zfill(6)
    mapping_df['gvkey'] = mapping_df['gvkey'].astype(str).str.zfill(6)
    mapping_df['permno'] = mapping_df['permno'].astype(int)
    daily_df['permno'] = daily_df['permno'].astype(int)

    fundq_df['datadate'] = pd.to_datetime(fundq_df['datadate'])
    fundq_df['rdq'] = pd.to_datetime(fundq_df['rdq'])
    daily_df['dlycaldt'] = pd.to_datetime(daily_df['dlycaldt'])
    mapping_df['linkdt'] = pd.to_datetime(mapping_df['linkdt'])
    mapping_df['linkenddt'] = pd.to_datetime(mapping_df['linkenddt'])

    # Drop restatements
    fundq_df = fundq_df.sort_values('rdq')
    fundq_df = fundq_df.drop_duplicates(subset=['gvkey', 'datadate'], keep='first')

    # Map gvkey to permno
    fundq_merged = fundq_df.merge(mapping_df, on='gvkey', how='inner')
    fundq_merged = fundq_merged[
        (fundq_merged['rdq'] >= fundq_merged['linkdt']) &
        ((fundq_merged['rdq'] <= fundq_merged['linkenddt']) | fundq_merged['linkenddt'].isna())
    ]

    if fundq_cols:
        fundq_cols = list(dict.fromkeys(fundq_cols + ['permno', 'rdq']))
        fundq_merged = fundq_merged[fundq_cols]

    # Prepare merge tasks
    all_permnos = set(daily_df['permno']) & set(fundq_merged['permno'])
    tasks = []
    for permno in all_permnos:
        df_daily = daily_df[daily_df['permno'] == permno].copy()
        df_fundq = fundq_merged[fundq_merged['permno'] == permno].copy()
        tasks.append((permno, df_daily, df_fundq, max_days_back))

    # Parallel execution using threads
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(merge_single_permno, tasks))

    merged_all = pd.concat(results, ignore_index=True)

    # Final column selection
    final_cols = []
    if fundq_cols:
        final_cols += fundq_cols
    if daily_cols:
        final_cols += daily_cols
    final_cols = list(dict.fromkeys(final_cols))

    return merged_all[final_cols]
