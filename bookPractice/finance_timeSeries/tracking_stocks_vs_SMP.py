import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

portfolio = [('KO', 8), ('AMZN', 4), ('GOOG', 4)]

dfs = []
for ticker_symbol, quantity in portfolio:
    tick = yf.Ticker(ticker_symbol)
    df = tick.history(start="2022-11-12")[['Close']]
    df = df.reset_index()
    df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')
    df = df[['Date_str', 'Close']]
    df = df.rename(columns={'Close': ticker_symbol})
    df[ticker_symbol] = df[ticker_symbol] * quantity
    dfs.append(df)

portfolio_df = reduce(lambda left, right: pd.merge(left, right, on='Date_str', how='outer'), dfs)
portfolio_df = portfolio_df.fillna(0)
portfolio_df['total'] = portfolio_df[[t[0] for t in portfolio]].sum(axis=1)



# S&P comp
tick = yf.Ticker('^GSPC')
df = tick.history(start="2022-11-12")
df_reset = df.reset_index()
df_reset['Date_str'] = df_reset['Date'].dt.strftime('%Y-%m-%d')
# Scale S&P 500 close prices to match portfolio starting point (roughly)
df_reset['Close_scaled'] = df_reset['Close'] * 0.31


portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date_str'])


#printing the starts and ends for both
print("Portfolio total on first date ({}): ${:.2f}".format(
    portfolio_df['Date'].iloc[0].strftime('%Y-%m-%d'),
    portfolio_df['total'].iloc[0]
))

print("Portfolio total on last date ({}): ${:.2f}".format(
    portfolio_df['Date'].iloc[-1].strftime('%Y-%m-%d'),
    portfolio_df['total'].iloc[-1]
))

print("S&P 500 close on first date ({}): ${:.2f}".format(
    df_reset['Date'].iloc[0].strftime('%Y-%m-%d'),
    df_reset['Close_scaled'].iloc[0]
))

print("S&P 500 close on last date ({}): ${:.2f}".format(
    df_reset['Date'].iloc[-1].strftime('%Y-%m-%d'),
    df_reset['Close_scaled'].iloc[-1]
))

plt.figure(figsize=(12, 6))

plt.plot(df_reset['Date'], portfolio_df['total'], label='portfolio', color='blue')
plt.plot(df_reset['Date'], df_reset['Close_scaled'], label='S&P 500', color='red')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.title('my portfolio VS. S&P 500')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


