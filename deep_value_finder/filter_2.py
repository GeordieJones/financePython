import pandas as pd
import os
import yfinance as yf
import time
'''
volitilities = 'nasdaq_volatilities.csv'

vols = pd.read_csv(volitilities)
vols['volatility'] = pd.to_numeric(vols['volatility'], errors='coerce')
acceptable_vols = []

# Thresholds
min_vol = 0.01
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
'''
def get_1y_return(symbol):
        data = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
        if data.empty or 'Close' not in data.columns:
            print(f"No data for {symbol}, returning none.")
            return None
        close_prices = data['Close']
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0]

        start_price = close_prices.iloc[0]
        end_price = close_prices.iloc[-1]
        return_pct = (end_price - start_price) / start_price
        return return_pct



def Y_1_filter():
    stocks = pd.read_csv('acceptable_volatilities.csv')
    min_ret = -0.3
    max_ret = 1.5
    acceptable_stocks = []

    
    for idx, row in stocks.iterrows():
        symbol = row['symbol']
        return_1y = get_1y_return(symbol)

        if return_1y is None:
            print(f"No 1 year return data for {symbol}, skipping.")
            continue

        if return_1y < min_ret:
            print(f"Too low return found: {symbol} with return of {return_1y:.2f}")
        elif return_1y > max_ret:
            print(f"Too high return found: {symbol} with return of {return_1y:.2f}")
        else:
            acceptable_stocks.append([symbol, return_1y])
            print(f"Acceptable return found: {symbol} with return of {return_1y:.2f}")

        time.sleep(0.5)

    print (f"Number of acceptable tickers: {len(acceptable_stocks)}")
    return acceptable_stocks


def make_volume_list():
    vol_df = pd.read_csv('final_tickers.csv')
    volumes = []

    for idx, row in vol_df.iterrows():
        symbol = row['symbol']
        try:
            data = yf.download(symbol, period="10d", interval="1d", progress=False, auto_adjust=True)
            if data.empty or 'Volume' not in data.columns:
                print(f"No data for {symbol}, skipping.")
                continue
            volume = data['Volume']
            if isinstance(volume, pd.DataFrame):
                volume = volume.iloc[:, 0]
            volume = volume.mean()
            volumes.append([symbol, volume])
            print(f"Processed {symbol}: Average Volume = {volume:.2f}")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue
    
    volume_df = pd.DataFrame(volumes, columns=['symbol', 'average_volume'])
    volume_df = volume_df.sort_values(by='average_volume', ascending=False)
    volume_df = volume_df.head(1000)
    volume_df.to_csv('top_1000_stocks.csv', index=False)


make_volume_list()