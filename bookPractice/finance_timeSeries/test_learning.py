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
knowncash = 500
shares = 0
onhand = []
totalGuess = []
onhand_known = []
cashShare = []

for i in range(len(df)):
    sharecost = df_reset.at[i, 'Close']
    if i == 0:
        while knowncash >= sharecost:
            shares += 1
            knowncash -= sharecost
    else:
        prev_price = df_reset.at[i-1, 'Close']
        if sharecost > prev_price:
            while knowncash > sharecost:
                shares += 1
                knowncash -= sharecost
        else:
            knowncash += (shares*sharecost)
            shares = 0
    net_worth = knowncash + shares * sharecost
    onhand_known.append(knowncash)
    cashShare.append(net_worth)

df_reset['Cash'] = onhand_known
df_reset['Net Worth'] = cashShare


money = 0
cash = 500
hold_amount =100
for i in range(2, len(df)):
    sharecost = df_reset.at[i-1, 'Close']
    prev_price = df_reset.at[i-2, 'Close']
    org = cash
    if sharecost > prev_price:
        amount  = (int)((cash -hold_amount) / df_reset.at[i, 'Open'])
        profit = (amount * (df_reset.at[i, 'Close'] - df_reset.at[i, 'Open']))
        cash += profit
    guess_worth = cash
    onhand.append(cash - org)
    totalGuess.append(guess_worth)
#needed to add pading should be fixed later
df_reset['Profit'] = [None, None] + onhand
df_reset['Guess Worth'] = [None, None] + totalGuess
arr = df_reset[['Date_str', 'Close', 'Profit', 'Guess Worth' ,'Cash', 'Net Worth']].to_numpy()

print(arr[-5:])


plt.figure(figsize=(12, 6))


plt.plot(df_reset['Date'], df_reset['Close'], label='Close Price', color='blue')
plt.plot(df_reset['Date'], df_reset['Profit'], label='movement trading guess profit', color='green')
plt.plot(df_reset['Date'], df_reset['Net Worth'], label='known movement', color='red')
plt.plot(df_reset['Date'], df_reset['Guess Worth'], label='total guess money', color='purple')

plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.title('AAPL closing VS. reactive buy sell strategy')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
