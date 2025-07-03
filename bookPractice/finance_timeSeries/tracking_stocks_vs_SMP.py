import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

portfolio = [('KO', 8), ('AMZN', 4),('GOOG', 4)]
dfs = []

portfolioWnivida = [('KO', 8), ('AMZN', 4),('GOOG', 4), ('NVDA', 10)]
dfsWnivida = []
for ticker_symbol, quantity in portfolioWnivida:
    tick = yf.Ticker(ticker_symbol)
    df = tick.history(start="2022-11-12")[['Close']]
    df = df.reset_index()
    df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')
    df = df[['Date_str', 'Close']]
    df = df.rename(columns={'Close': ticker_symbol})
    df[ticker_symbol] = df[ticker_symbol] * quantity
    dfsWnivida.append(df)

portfolio_Nvida_df = reduce(lambda left, right: pd.merge(left, right, on='Date_str', how='outer'), dfsWnivida)
#portfolio_Nvida_df = portfolio_Nvida_df.fillna(method='ffill')
portfolio_Nvida_df = portfolio_Nvida_df.ffill()
portfolio_Nvida_df['total'] = portfolio_Nvida_df[[t[0] for t in portfolioWnivida]].sum(axis=1)



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
#portfolio_df = portfolio_df.fillna(method='ffill')
portfolio_df = portfolio_df.ffill()
portfolio_df['total'] = portfolio_df[[t[0] for t in portfolio]].sum(axis=1)



# S&P comp
tick = yf.Ticker('^GSPC')
df = tick.history(start="2022-11-12")
df_reset = df.reset_index()
df_reset['Date_str'] = df_reset['Date'].dt.strftime('%Y-%m-%d')
# Scale S&P 500 close prices to match portfolio starting point (roughly)
df_reset['Close_scaled'] = df_reset['Close'] * 0.31



comparison_df = pd.merge(
    portfolio_df[['Date_str', 'total']],
    df_reset[['Date_str', 'Close_scaled']],
    #portfolio_Nvida_df[['Date_str', 'total']],
    on='Date_str',
    how='inner'
)

comparison_df = pd.merge(
    comparison_df,
    portfolio_Nvida_df[['Date_str', 'total']].rename(columns={'total': 'total_with_nvda'}),
    on='Date_str',
    how='inner'
)


# Calculate difference on matched dates
comparison_df['difference'] = comparison_df['total'] - comparison_df['Close_scaled']

# Convert Date_str to datetime for plotting
comparison_df['Date'] = pd.to_datetime(comparison_df['Date_str'])

portfolio_Nvida_df['Date']= pd.to_datetime(portfolio_Nvida_df['Date_str'])


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

plt.plot(comparison_df['Date'], comparison_df['total'], label='Portfolio', color='blue')
plt.plot(comparison_df['Date'], comparison_df['Close_scaled'], label='S&P 500 (scaled)', color='red')
plt.plot(comparison_df['Date'], comparison_df['difference'], label='Difference', color='green')
plt.plot(comparison_df['Date'], comparison_df['total_with_nvda'], label='Portfolio w/ NVIDIA', color='purple')

plt.xlabel('Date')
plt.ylabel('Value')
plt.title('My Portfolio vs. S&P 500')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



