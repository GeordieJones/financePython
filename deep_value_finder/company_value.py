import requests
import pandas as pd
import json
import requests
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


API_KEYS = ["9PYBJ0V552GDDNLF", "UA4UENTKVXQ3KUVI", "1WL2VIGXHPNJ8QBN","O0UN1OLLKQV2VEPZ"]
tickers_from_file = pd.read_csv('top_1000_stocks.csv')
tickers = tickers_from_file['symbol'].tolist()
symbol = 'AAPL'
function = 'OVERVIEW'


def fetch_overview(args):
    ticker, key = args
    max_retries = 3
    retries = 0

    while retries < max_retries:
        url = f"https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={key}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if 'Note' in data:
                    print(f"[{key}] Rate limit hit for {ticker}. Sleeping 60s... (retry {retries + 1}/{max_retries})")
                    retries += 1
                    time.sleep(60)
                    continue
                if 'Symbol' not in data:
                    print(f"No symbol found in response for {ticker}")
                    return None
                time.sleep(12)
                return data
            else:
                print(f"HTTP {response.status_code} for {ticker}")
                return None
        except Exception as e:
            print(f"Error for {ticker}: {e}")
            return None

    print(f"Max retries exceeded for {ticker}")
    return None

def get_all_overviews():
    args_list = [(ticker, API_KEYS[i % len(API_KEYS)]) for i, ticker in enumerate(tickers)]

    with Pool(processes=4) as pool:
        results = list(tqdm(pool.imap_unordered(fetch_overview, args_list), total=len(args_list)))

    valid_results = [r for r in results if r]
    df = pd.DataFrame(valid_results)
    return df

if __name__ == '__main__':
    df = get_all_overviews()
    df.to_csv("company_overviews.csv", index=False)
    print("Finished. Data saved to company_overviews.csv")
