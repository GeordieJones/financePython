import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import sentiment_finder as sf

tickers = ['AAPL', 'KO', 'MSFT', 'AMZN', 'NVDA', 'JPM', 'GOOG',"PLTR", "NFLX", "ARKK", "SOFI", "MCD", "PG", "JNJ", "XOM", "BRK-B"]
day = datetime(2025, 7, 22)

def plot_sentiment_v_price(tickers, date):
    next_day = date + timedelta(days=1)
    prices = []
    sentiment = []
    for ticker in tickers:
        # Fetch stock data
        prices.append(yf.download(ticker, start=date, end=next_day))
        
        ticker_obj = yf.Ticker(ticker)
        short_name = ticker_obj.info.get("shortName", ticker) 
        keywords = [ticker, short_name]

        sentiment.append(sf.get_stock_sentiment(ticker, keywords, days_in_past=1, start_day=date))

    plt.scatter(prices, sentiment, label=ticker)
    plt.xlabel('Stock Price')
    plt.ylabel('Sentiment') 
    plt.title(f'Sentiment vs Price on {date.strftime("%Y-%m-%d")}')
    plt.legend()
    plt.show()

plot_sentiment_v_price(tickers, day)

