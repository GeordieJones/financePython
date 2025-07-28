import pandas as pd
import os
import yfinance as yf
import time

def filter_tickers():
    
    check = 'nasdaq_tickers.csv'
    available_tickers = 'avalible_tickers.csv'
    combine = 'filtered_tickers.csv'


    nasdaq_tickers = pd.read_csv(check)
    available = pd.read_csv(available_tickers)

    filtered = available[
        (available['exchange'] == 'NASDAQ') &
        (available['status'] == 'Active') &
        (available['assetType'].isin(['Stock', 'ETF']))
    ]

    print(f"Number of filtered tickers: {len(filtered)}")

    # Write to CSV
    filtered.to_csv(combine, index=False)


def find_volatilities():
    tickers_df = pd.read_csv('filtered_tickers.csv')
    symbols = tickers_df['symbol'].tolist()

    stock_vols = []

    for i, symbol in enumerate(symbols):
        print(f"Processing {i+1}/{len(symbols)}: {symbol}...", end=' ')
        try:
            data = yf.download(symbol, period="60d", interval="1d", progress=False, auto_adjust=True)

            # Skip if no data or if Close column is missing
            if data.empty or 'Close' not in data.columns:
                print("No data, skipping.")
                continue

            # Compute percent change and volatility
            pct_changes = data['Close'].pct_change().dropna()
            volatility = pct_changes.std()
            if isinstance(volatility, pd.Series):
                volatility = volatility.mean()

            stock_vols.append([symbol, volatility])
            print(f"Volatility: {volatility:.2f}%")

        except Exception as e:
            print(f"Failed: {e}")
        
        time.sleep(0.5)  # avoid rate limiting

    # Save to CSV
    result_df = pd.DataFrame(stock_vols, columns=['symbol', 'volatility'])
    result_df.to_csv('nasdaq_volatilities.csv', index=False)
    print("\nDone. Saved volatilities to 'nasdaq_volatilities.csv'.")

find_volatilities()






    

