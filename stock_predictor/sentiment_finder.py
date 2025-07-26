import finnhub
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
today = datetime.today()
minus_time = today - timedelta(days=180)


# Setup client
finnhub_client = finnhub.Client(api_key='d21a131r01qkdupigrm0d21a131r01qkdupigrmg')

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        print("Loading model for the first time...")
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


def get_stock_sentiment(symbol, key_words, days_in_past = 1, start_day = datetime.today()):
    load_model()
    to_date = start_day
    from_date = start_day - timedelta(days=days_in_past)
    news = finnhub_client.company_news(symbol, _from=from_date.strftime('%Y-%m-%d'), to=to_date.strftime('%Y-%m-%d'))
    company_name_keywords = key_words
    filtered_news = [
        article for article in news
        if any(keyword.lower() in (article['headline'] + article['summary']).lower() for keyword in company_name_keywords)
    ]

    filtered_news = filtered_news[:5]

    sentences = [news['summary'] for news in filtered_news]
    if not sentences:
        print(f"No news found for {symbol} on {from_date.strftime('%Y-%m-%d')}")
        return np.array([0, 0, 0])
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    score_array = [0 , 0, 0]#'positive', 'negative', 'neutral'

    probabilities = torch.softmax(logits, dim=1).numpy()
    labels = ['positive', 'negative', 'neutral']
    for i, sentence in enumerate(sentences):
        probs = probabilities[i]
        pred_label = labels[np.argmax(probs)]
        for j in range(3):
            score_array[j] += probs[j]
    scores = np.array(score_array) / np.sum(score_array)
    #print(f'{symbol}\'s overall sentiment: {labels[np.argmax(score_array)]}')
    #print(f'{symbol}\'s values sentiment(positive, negative, neutral): {scores}')
    time.sleep(1.1)
    return scores

def sentiment_history(ticker, start_date, end_date, key_words):
    date_range = pd.date_range(start=start_date, end=end_date, freq='2D')
    sentiment_scores = []

    for date in tqdm(date_range, desc=f'Processing {ticker} sentiment'):
        sentiment = get_stock_sentiment(ticker, key_words, days_in_past=1, start_day=date)
        sentiment_scores.append({
            'date': date,
            'sentiment': sentiment
        })

    return pd.DataFrame(sentiment_scores)

def dateframe_sentiment(tickers, start_date, end_date, key_words):
    date_range = pd.date_range(start=start_date, end=end_date)
    sentiment_dict = {date: {} for date in date_range}

    for ticker in tickers:
        ticker_sentiments = sentiment_history(ticker, start_date, end_date, key_words)
        for _, row in ticker_sentiments.iterrows():
            sentiment_dict[row['date']][ticker] = row['sentiment']

    sentiment_df = pd.DataFrame.from_dict(sentiment_dict, orient='index')
    sentiment_df.index.name = 'date'
    sentiment_df = sentiment_df.sort_index()
    return sentiment_df

def plot_sentiment(sentiment_df):
    for column in sentiment_df.columns:
        sentiment_df[column] = sentiment_df[column].apply(lambda x: x[0] - x[1])
    ax = sentiment_df.plot(figsize=(12, 6))
    plt.title('Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    ax.figure.autofmt_xdate()
    plt.show()

def process_stocks(stocks):
# Loop through each stock and process
    for ticker, keywords in tqdm(stocks.items(), desc="Processing all stocks"):
        print(f"\nProcessing {ticker} sentiment data...")
        sentiment_df = dateframe_sentiment([ticker], minus_time, today, keywords)
        filename = f'{ticker.lower().replace("-", "").replace("^", "")}_sentiment.pkl'
        sentiment_df.to_pickle(filename)
        print(f"Saved {ticker} sentiment to {filename}")


def combine_sentiments_from_files(stock_list):
    dfs = []

    for ticker in stock_list:
        filename = f'{ticker.lower().replace("-", "").replace("^", "")}_sentiment.pkl'
        if not os.path.exists(filename):
            print(f"File {filename} not found. Skipping {ticker}.")
            continue

        df = pd.read_pickle(filename)

        # If empty or missing values
        if df.empty or ticker not in df.columns:
            print(f"Data for {ticker} is empty or improperly formatted.")
            continue

        df[ticker] = df[ticker].apply(lambda x: x[0] - x[1])
        df = df[[ticker]]
        dfs.append(df)

    if not dfs:
        raise ValueError("No valid sentiment data found.")

    combined_df = pd.concat(dfs, axis=1, join='outer')
    combined_df.index.name = 'date'
    combined_df = combined_df.sort_index()
    return combined_df


def plot_combined_sentiments(combined_df):
    ax = combined_df.plot(figsize=(14, 7), linewidth=1.8, colormap='tab20')  # Use distinct colors
    plt.title('Stock Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score (Positive - Negative)')
    plt.legend(title="Stock Ticker", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.figure.autofmt_xdate()
    plt.tight_layout()
    plt.grid(True)
    plt.show()


stocks = {
    'MSFT': ['MSFT', 'Microsoft'],
    #'AAPL': ['AAPL', 'Apple'],
    'GOOG': ['GOOGL', 'Google', 'Alphabet'],
    'AMZN': ['AMZN', 'Amazon'],
    'TSLA': ['TSLA', 'Tesla'],
    'KO': ['KO', 'Coca-Cola'],
    'NVDA': ['NVDA', 'Nvidia'],
    'JPM': ['JPM', 'JPMorgan Chase'],
    'PLTR': ['PLTR', 'Palantir'],
    'NFLX': ['NFLX', 'Netflix'],
    'MCD': ['MCD', 'McDonald\'s', 'McDonalds', 'McDonalds Corporation'],
    'SPY': ['S&P 500', '500', 'SP500', 'S&P'],
    'JNJ': ['JNJ', 'Johnson & Johnson', 'healthcare', 'pharmaceuticals'],
    'XOM': ['XOM', 'ExxonMobil','Exxon', 'Exxon Mobil', 'oil', 'gas'],
    'META': ['META', 'Meta Platforms', 'Facebook', 'Instagram', 'WhatsApp', 'Zuckerberg']
}


combined_df = combine_sentiments_from_files(stocks.keys())
plot_combined_sentiments(combined_df)
