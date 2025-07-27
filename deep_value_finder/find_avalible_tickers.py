import pandas as pd
import requests
import time
import threading
import os
from queue import Queue
from datetime import datetime

'''# Step 1: Fetch the data
url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
response = requests.get(url)

# Step 2: Parse it
data = response.text
df = pd.read_csv(StringIO(data), sep='|')

# Step 3: Filter only NYSE tickers
nyse_tickers = df[df['Exchange'] == 'N']['ACT Symbol'].tolist()

pd.DataFrame(nyse_tickers, columns=["Ticker"]).to_csv("nyse_tickers.csv", index=False)
print(f"Saved {len(nyse_tickers)} NYSE tickers to nyse_tickers.csv")

print(f"Total NYSE tickers: {len(nyse_tickers)}")'''


API_KEYS = ["9PYBJ0V552GDDNLF", "UA4UENTKVXQ3KUVI", "1WL2VIGXHPNJ8QBN","O0UN1OLLKQV2VEPZ"]
valid_tickers = []
INPUT_FILE = "nyse_tickers.csv"
SAVE_FILE = "validated_tickers.csv"
REQUESTS_PER_MIN = 5

# === GLOBAL LOCK FOR CSV SAVING ===
lock = threading.Lock()

# === Load ticker list ===
all_tickers = pd.read_csv(INPUT_FILE)['Ticker'].tolist()
# === Resume if previously saved ===
if os.path.exists(SAVE_FILE):
    existing_df = pd.read_csv(SAVE_FILE)
    checked = set(existing_df['Ticker'])
else:
    existing_df = pd.DataFrame(columns=["Ticker", "Valid"])
    checked = set()

# === Filter only unchecked tickers ===
unchecked_tickers = [t for t in all_tickers if t not in checked]

# === Shared queue for threads ===
ticker_queue = Queue()
for ticker in unchecked_tickers:
    ticker_queue.put(ticker)


def worker(api_key: str, thread_id: int):
    calls_made = 0
    start_time = time.time()

    while not ticker_queue.empty():
        if calls_made >= REQUESTS_PER_MIN:
            elapsed = time.time() - start_time
            if elapsed < 60:
                time.sleep(60 - elapsed)
            calls_made = 0
            start_time = time.time()

        try:
            symbol = ticker_queue.get_nowait()
        except:
            break

        try:
            url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}"
            r = requests.get(url, timeout=10)
            data = r.json()
            is_valid = 'Name' in data

            row = pd.DataFrame([{"Ticker": symbol, "Valid": is_valid}])

            # Save result safely
            with lock:
                row.to_csv(SAVE_FILE, mode='a', header=not os.path.exists(SAVE_FILE), index=False)

            print(f"[Thread {thread_id}] {symbol} {'✅' if is_valid else '❌'}")

        except Exception as e:
            print(f"[Thread {thread_id}] {symbol} ❌ Request failed: {e}")

        calls_made += 1
        time.sleep(60 / REQUESTS_PER_MIN)  # ~12 sec per request


# === Start threads ===
threads = []
for i, key in enumerate(API_KEYS):
    t = threading.Thread(target=worker, args=(key, i + 1))
    t.start()
    threads.append(t)

# === Wait for threads to finish ===
for t in threads:
    t.join()

print(f"\n✅ Done! Saved to {SAVE_FILE}")