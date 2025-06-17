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

print(f"Data shape: {data.shape}")
print("First 5 rows:")
print(data[:5])

plt.figure(figsize=(10, 6))
plt.plot(df_reset['Date'], df_reset['Close'], label='Close Price')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.title('AAPL Closing Prices Last 6 Months')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
