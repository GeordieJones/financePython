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

volitilities = 'nasdaq_volatilities.csv'

vols = pd.read_csv(volitilities)
vols['volatility'] = pd.to_numeric(vols['volatility'], errors='coerce')
acceptable_vols = []

# Thresholds
min_vol = 0.005
max_vol_above_mean = 0.04
mean_vol = vols['volatility'].mean()
max_vol = mean_vol + max_vol_above_mean

for idx, row in vols.iterrows():
    symbol = row['symbol']
    volatility = row['volatility']

    if volatility < min_vol:
        print(f"Too low volatility found: {symbol} with volatility {volatility:.2f}")
    elif volatility > max_vol:
        print(f"Too high volatility found: {symbol} with volatility {volatility:.2f}")
    elif pd.isna(volatility):
        print(f"No volatility data for {symbol}, skipping.")
    else:
        acceptable_vols.append([symbol, volatility])
        print(f"Acceptable volatility found: {symbol} with volatility {volatility:.2f}")

print (f"Number of acceptable tickers: {len(acceptable_vols)}")
result_df = pd.DataFrame(acceptable_vols, columns=['symbol', 'volatility'])
result_df.to_csv('acceptable_volatilities.csv', index=False)

def filter_out():
    df = pd.read_csv('acceptable_volatilities.csv')
    comp = pd.read_csv('avalible_tickers.csv')

    comp_stocks = comp[comp['assetType'] == 'Stock']
    comp_stocks = comp_stocks['symbol'].tolist()

    corrected_acceptable = []
    for idx, row in df.iterrows():
        symbol = row['symbol']
        if symbol in comp_stocks:
            corrected_acceptable.append([symbol, row['volatility']])
        else:
            print(f"Symbol {symbol} not found in available tickers, skipping.")
    print(f"Number of filtered tickers: {len(corrected_acceptable)}")
    result_df = pd.DataFrame(corrected_acceptable, columns=['symbol', 'volatility'])
    result_df.to_csv('acceptable_volatilities.csv', index=False)



def filter_and_rank_stocks():
    tickers_df = pd.read_csv('acceptable_volatilities.csv')
    min_ret = -0.3
    max_ret = 1.5
    results = []

    for idx, row in tickers_df.iterrows():
        symbol = row['symbol']
        try:
            data = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
            if data.empty or 'Close' not in data.columns or 'Volume' not in data.columns:
                print(f"No data for {symbol}, skipping.")
                continue

            # --- Calculate 1-Year Return ---
            close_prices = data['Close']
            if isinstance(close_prices, pd.DataFrame):
                close_prices = close_prices.iloc[:, 0]

            start_price = close_prices.iloc[0]
            end_price = close_prices.iloc[-1]
            return_1y = (end_price - start_price) / start_price

            if return_1y < min_ret:
                print(f"Too low return found: {symbol} with return of {return_1y:.2f}")
                continue
            elif return_1y > max_ret:
                print(f"Too high return found: {symbol} with return of {return_1y:.2f}")
                continue

            # --- Calculate 10-Day Average Volume ---
            recent_data = data.tail(10)
            volume = recent_data['Volume']
            if isinstance(volume, pd.DataFrame):
                volume = volume.iloc[:, 0]
            average_volume = volume.mean()

            results.append([symbol, return_1y, average_volume])
            print(f"Accepted {symbol}: Return = {return_1y:.2f}, Avg Volume = {average_volume:.2f}")

        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue

        time.sleep(0.1)  # optional to avoid rate limits

    df = pd.DataFrame(results, columns=['symbol', '1y_return', 'average_volume'])
    df = df.sort_values(by='average_volume', ascending=False).head(1000)
    df.to_csv('top_1000_stocks.csv', index=False)
    print(f"Saved top {len(df)} stocks to top_1000_stocks.csv")
    return df


filter_and_rank_stocks()

