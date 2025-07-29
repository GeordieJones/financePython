import requests
import pandas as pd
from tqdm import tqdm
import yfinance as yf
import time
from numbers import Number


tickers_from_file = pd.read_csv('top_1000_stocks.csv')
tickers = tickers_from_file['symbol'].tolist()

financials = []
index_labels = []

for ticker in tqdm(tickers):
    try:
        fin_df = yf.Ticker(ticker).financials
        if fin_df.empty:
            continue
        
        # Transpose and grab the latest row (first one is most recent)
        latest = fin_df.T.iloc[0]
        latest.name = ticker  # Set ticker as index
        financials.append(latest)
        index_labels.append(ticker)
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
    time.sleep(0.1)

# Combine into a DataFrame with tickers as rows
financials_df = pd.DataFrame(financials, index=index_labels)

# Drop rows and columns that are all NaNs
financials_df.dropna(axis=0, how='all', inplace=True)  # Drop tickers with no valid data
financials_df.dropna(axis=1, how='all', inplace=True)  # Drop metrics that are empty for all tickers

# Sort by ticker symbol
financials_df.sort_index(inplace=True)

# Save cleaned data
financials_df.to_pickle('latest_data.pkl')