import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

ticker = yf.Ticker("AMZN")
exp_date = "2025-07-18"

opt = ticker.option_chain(exp_date)
calls = opt.calls

print(calls[['ask', 'strike', 'lastPrice', 'change','impliedVolatility']].head())

plt.figure(figsize=(10,6))
plt.plot(calls['strike'], calls['lastPrice'], label='Last Price', marker='o')
plt.plot(calls['strike'], calls['ask'], label='Ask Price', marker='x')

plt.xlabel('Strike Price')
plt.ylabel('Option Price (USD)')
plt.title(f'AMZN Call Options Prices - Expiration {exp_date}')
plt.legend()
plt.grid(True)
plt.show()