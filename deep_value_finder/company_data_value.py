import requests
import pandas as pd
from tqdm import tqdm
import yfinance as yf
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


def fetch_financials():
    tickers_from_file = pd.read_csv('top_1000_stocks.csv')
    tickers = tickers_from_file['symbol'].tolist()

    financials_yearly = {i: [] for i in range(5)}
    tickers_yearly = {i: [] for i in range(5)}


    for ticker in tqdm(tickers,desc = f"Processing tickers"):
        try:
            fin_df = yf.Ticker(ticker).financials
            if fin_df.empty:
                continue

            fin_df = fin_df.T.sort_index(ascending=False)
                
                # Transpose and grab the latest row (first one is most recent)
            for year_i in range(5):
                if len(fin_df) > year_i:
                    row = fin_df.iloc[year_i]
                    row.name = ticker
                    financials_yearly[year_i].append(row)
                    tickers_yearly[year_i].append(ticker)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
        time.sleep(0.1)

    columns_to_drop = [
        'Tax Effect Of Unusual Items', 'Total Unusual Items Excluding Goodwill',
        'Net Income From Continuing Operation Net Minority Interest',
        'Reconciled Depreciation', 'Reconciled Cost Of Revenue',
        'Net Interest Income', 'Net Income From Continuing And Discontinued Operation',
        'Diluted Average Shares', 'Basic Average Shares',
        'Basic EPS', 'Diluted NI Availto Com Stockholders',
        'Net Income Common Stockholders', 'Net Income Including Noncontrolling Interests',
        'Net Income Continuous Operations', 'Other Non Operating Income Expenses',
        'Special Income Charges', 'Net Non Operating Interest Income Expense',
        'Interest Expense Non Operating', 'Interest Income Non Operating',
        'Operating Revenue', 'General And Administrative Expense', 'Other Gand A'
    ]

    threshold = 0.5
    for year_i in range(5):
        df = pd.DataFrame(financials_yearly[year_i], index=tickers_yearly[year_i])
        
        # Drop rows and columns with all NaNs
        df.dropna(axis=0, how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        
        # Drop columns with too many NaNs
        min_non_na = int(threshold * len(df))
        df = df.dropna(axis=1, thresh=min_non_na)
        
        # Drop manually curated redundant columns if present
        cols_to_drop = [c for c in columns_to_drop if c in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)
        
        df.sort_index(inplace=True)
        
        df.to_pickle(f'year{year_i}_data.pkl')  


def add_close_in_9mo():
    fiscal_years = {
        0: datetime(2024, 9, 30),
        1: datetime(2023, 9, 30),
        2: datetime(2022, 9, 30),
        3: datetime(2021, 9, 30)
    }

    for year_i in range(4):
        df = pd.read_pickle(f'year{year_i}_data.pkl')
        close_prices = []

        target_date = fiscal_years[year_i] + relativedelta(months=+9)

        fail_count = 0
        total_count = len(df)

        for ticker in tqdm(df.index, desc=f"Getting close prices for year {year_i}"):
            try:
                hist = yf.Ticker(ticker).history(start=target_date - timedelta(days=7), end=target_date + timedelta(days=7))
                if not hist.empty:
                    close_price = hist['Close'].dropna().iloc[0] if not hist['Close'].dropna().empty else None
                else:
                    fail_count += 1
                    close_price = None
            except Exception as e:
                print(f"{ticker} failed: {e}")
                close_price = None

            close_prices.append(close_price)

        df['Close_9mo_after'] = close_prices
        df.to_pickle(f'year{year_i}_data_labeled.pkl')
        print(f"\n[Year {year_i}] Missed close price for {fail_count}/{total_count} stocks ({100*fail_count/total_count:.1f}%)\n")


def remove_missing_close_9mo():
    for year_i in range(4):
        df = pd.read_pickle(f'year{year_i}_data_labeled.pkl')
        # Drop rows where 'Close_9mo_after' is None or NaN
        df_clean = df.dropna(subset=['Close_9mo_after'])
        print(f"[Year {year_i}] Dropped {len(df) - len(df_clean)} tickers without Close_9mo_after price.")
        df_clean.to_pickle(f'year{year_i}_data_labeled.pkl')

remove_missing_close_9mo()