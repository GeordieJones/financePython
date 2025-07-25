import finnhub
from datetime import datetime, timedelta
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import pandas as pd
today = datetime.today()
minus_time = today - timedelta(days=1)


# Setup client
finnhub_client = finnhub.Client(api_key='d21a131r01qkdupigrm0d21a131r01qkdupigrmg')

# Load FinBERT tokenizer and model
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)



def get_stock_sentiment(symbol, key_words, days_in_past = 1, start_day = datetime.today()):
    from_date = start_day
    to_date = start_day - timedelta(days=days_in_past)
    news = finnhub_client.company_news(symbol, _from=from_date.strftime('%Y-%m-%d'), to=to_date.strftime('%Y-%m-%d'))
    company_name_keywords = key_words
    filtered_news = [
        article for article in news
        if any(keyword.lower() in (article['headline'] + article['summary']).lower() for keyword in company_name_keywords)
    ]

    filtered_news = filtered_news[:5]

    sentences = [news['summary'] for news in filtered_news]
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.softmax(logits, dim=1).numpy()
    labels = ['neutral', 'positive', 'negative']
    for i, sentence in enumerate(sentences):
        score_array = [0 , 0, 0]#neutral, positive, negative
        probs = probabilities[i]
        pred_label = labels[np.argmax(probs)]
        for i in range(3):
            score_array[i] += probs[i]

    print(f'{symbol}\'s overall sentiment: {labels[np.argmax(score_array)]}')

    return labels[np.argmax(score_array)]

def sentiment_history(ticker, start_date, end_date, key_words):
    date_range = pd.date_range(start=start_date, end=end_date)
    sentiment_scores = []

    for date in date_range:
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
