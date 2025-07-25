import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import sentiment_finder as sf

tickers = ['AAPL', 'KO', 'MSFT', 'AMZN', 'NVDA', 'JPM', 'GOOG',"PLTR", "NFLX", "ARKK", "SOFI", "MCD", "PG", "JNJ", "XOM", "BRK-B",'META']
day = datetime(2025, 7, 22)

def plot_sentiment_v_price(tickers, date):
    next_day = date + timedelta(days=1)
    prices = []
    sentiment = []
    for ticker in tickers:
        # Fetch stock data
        df = yf.download(ticker, start=date, end=next_day)
        prices.append(float(df['Close'].iloc[0]))


        ticker_obj = yf.Ticker(ticker)
        short_name = ticker_obj.info.get("shortName", ticker) 
        keywords = [ticker, short_name]

        sentiment_score = (sf.get_stock_sentiment(ticker, keywords, days_in_past=1, start_day=date))
        sentiment_score = sentiment_score[0]-sentiment_score[1]
        sentiment.append(sentiment_score)

    plt.scatter(prices, sentiment)
    for i, ticker in enumerate(tickers):
        plt.annotate(ticker, (prices[i], sentiment[i]), textcoords="offset points", xytext=(5,5), ha='left')
    plt.xlabel('Stock Price')
    plt.ylabel('Sentiment') 
    plt.title(f'Sentiment vs Price on {date.strftime("%Y-%m-%d")}')
    plt.show()

plot_sentiment_v_price(tickers, day)

