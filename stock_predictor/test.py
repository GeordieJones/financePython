import os
import pandas as pd



def get_sentiment_data(sentiment_dir="sentiment_data"):
    files = os.listdir(sentiment_dir)
    sentiment_files = [f for f in files if f.endswith('_sentiment.pkl')]
    print("Sentiment files found:", sentiment_files)

    dfs = []

    for file in sentiment_files:
        ticker = file.split('_')[0]
        filepath = os.path.join(sentiment_dir, file)
        df = pd.read_pickle(filepath)

        if 'date' in df.columns:
            df = df.set_index('date')

        df.index = pd.to_datetime(df.index)
        u_ticker = ticker.upper()
        # Unpack list column into 3 columns
        if u_ticker in df.columns:
            sentiment_lists = df[u_ticker].tolist()
            unpacked = pd.DataFrame(sentiment_lists, columns=[f"{u_ticker}_pos", f"{ticker}_neu", f"{ticker}_neg"], index=df.index)
            unpacked.index.name = 'date'
            unpacked = unpacked.groupby(pd.Grouper(freq='D')).mean()
            dfs.append(unpacked)
        else:
            print(f"Ticker {ticker} not found in columns: {df.columns}")

    return dfs


def load_and_process_sentiment(ticker):
    dfs = get_sentiment_data()
    for df in dfs:
        cols = [col for col in df.columns if col.startswith(ticker)]
        if cols:
            return df[cols]
    raise ValueError(f"No sentiment data found for ticker: {ticker}")

appl = load_and_process_sentiment('AAPL')
print(appl.head())
