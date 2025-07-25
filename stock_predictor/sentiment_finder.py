import finnhub
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
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
    time.sleep(1)
    return scores

def sentiment_history(ticker, start_date, end_date, key_words):
    date_range = pd.date_range(start=start_date, end=end_date)
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

sentiment_df = dateframe_sentiment(['AAPL'], minus_time, today, ['AAPL', 'Apple'])
sentiment_df.to_pickle('aapl_sentiment.pkl')
plot_sentiment(sentiment_df)

