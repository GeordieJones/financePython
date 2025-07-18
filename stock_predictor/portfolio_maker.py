import stock_guess as sg
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# could be any stocks you want to look at
hypothetical_portfolio = ['AAPL', 'KO', 'MSFT', 'AMZN', 'NVDA', 'JPM', 'GOOG']

def get_predictions(portfolio, lag_time = 'week'):
    next_week_prices = {}
    for ticker in portfolio:
        next_week_prices[ticker] = (sg.run_predict(ticker, start_check = '2018-01-01',end_check='2025-06-01', risk_free_rate=0, lag_time=lag_time))
    return next_week_prices

def create_comp(portfolio, prices):
    expected_returns = []
    for ticker in portfolio:
        close = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
        expected_returns.append((ticker, prices[ticker] - close, close))

    expected_returns.sort(key = lambda x: x[1], reverse=True)
    return expected_returns
    # now we have current prices and predicted in one week

def greedy_optimize(expected_returns, amount, limit=0.2):
    #simple greedy algorithm buys highest prediction until limit or money_limit
    amount_per_stock = []
    money = amount
    ovr = amount
    predicted_gain = 0
    for ticker, expected_return, price in expected_returns:
        current_amount = 0
        maxamount = int((ovr*limit)/price)
        while (money >= price) and (maxamount > current_amount):
            money -= price
            current_amount+=1
        predicted_gain += (expected_return * current_amount)
        stocks_gain = (expected_return * current_amount)
        amount_per_stock.append((ticker, current_amount, stocks_gain))
        

    return amount_per_stock, money, predicted_gain


def make_portfolio(portfolio, lag_time='week', amount = 10000, limit=0.2):
    predicted_prices = get_predictions(portfolio, lag_time = lag_time)
    expected_returns = create_comp(portfolio, predicted_prices)
    final, left_over, predicted_gain = greedy_optimize(expected_returns, amount, limit=limit)
    for ticker, total, gain in final:
        print(f'{ticker}: {total} shares \t contributed {round(gain, 2)}')
    print(f'leftover: {round(left_over, 2)} \n predicted gain: {round(predicted_gain,2)}')



make_portfolio(hypothetical_portfolio)

