import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

tick = yf.Ticker("AAPL")
df = tick.history(period="6mo")
df_reset = df.reset_index()

df_reset['Date_str'] = df_reset['Date'].dt.strftime('%Y-%m-%d')

data = df_reset[['Date_str', 'Open', 'High', 'Low', 'Close']].to_numpy()

#if you want to see the last part of the data
'''
print(f"Data shape: {data.shape}")
print("First 5 rows:")
print(data[:5])
'''
cash = df_reset.at[0, 'Close']
shares = 0
onhand = []
cashShare = []
for i in range(len(df)):
    sharecost = df_reset.at[i, 'Close']
    if i == 0:
        while cash >= sharecost:
            shares += 1
            cash -= sharecost
    else:
        prev_price = df_reset.at[i-1, 'Close']
        if sharecost > prev_price:
            while cash > sharecost:
                shares += 1
                cash -= sharecost
        else:
            cash += (shares*sharecost)
            shares = 0
    net_worth = cash + shares * sharecost
    onhand.append(cash)
    cashShare.append(net_worth)

df_reset['Cash'] = onhand
df_reset['Net Worth'] = cashShare

arr = df_reset[['Date_str', 'Close', 'Cash', 'Net Worth']].to_numpy()

print(arr[-5:])


plt.figure(figsize=(12, 6))


plt.plot(df_reset['Date'], df_reset['Close'], label='Close Price', color='blue')
plt.plot(df_reset['Date'], df_reset['Net Worth'], label='Net Worth', color='green')



plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.title('AAPL closing VS. reactive buy sell strategy')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
