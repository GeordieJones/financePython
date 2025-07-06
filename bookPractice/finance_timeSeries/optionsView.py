import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

ticker = yf.Ticker("AAPL")
exp_date = "2025-07-18"

opt = ticker.option_chain(exp_date)

print(opt.calls.head())
print(opt.puts.head())