import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import time

market_groups = {
    'standard': ['SPY'],
    "US Indexes": ['SPY', 'VOO', 'VGT'],
    "US Indicators": ['VPU', 'X', 'CAT', 'LMT', 'CLF','UNP', 'CSX', 'XOM', 'COP','HON'],
    "US Bonds": ['VGLT', 'IEF', 'JNK', 'TIP', 'TIP'],
    "India": ['INFY','HDB','IBN','TCS.NS', 'RELIANCE.NS','LT.NS', 'HINDUNILVR.NS', 'BHARTIARTL.NS', 'MARUTI.NS', 'AXISBANK.NS' ],
    "Japan": ['TM', 'NSANY', 'SMFG', 'SONY'],
    "China": ['FXI', 'GXC', 'JD', 'PDD', 'NTES', 'BIDU', 'TCEHY', 'BABA', 'MOMO', 'LI', 'NIO'],
    "Germany": ['EWG', '^GDAXI', 'SAP', 'ALV', 'BAYRY', 'MBGAF', 'VWAGY'],
    "UK": ['BP','HSBC', 'UL', 'GSK'],
    "EU": ['EZU', 'FEZ', 'IEV'],
    "Global": ['VT','VEU']
}

def plot_movements():
    all_markets = []
    plt.figure(figsize=(14, 8))
    for group_name, tickers in tqdm(market_groups.items(), desc="Downloading & Processing"):
        try:
            df = yf.download(tickers,period="10y", interval="1mo", progress=False, auto_adjust=True)['Close']

            if isinstance(df, pd.Series):
                df = df.to_frame()
            # Drop columns with all NaNs (i.e., failed tickers)
            df.dropna(axis=1, how='all', inplace=True)

            # Warn about any failed tickers
            missing_tickers = [ticker for ticker in tickers if ticker not in df.columns]
            if missing_tickers:
                print(f"⚠️  {group_name}: Failed to get data for: {missing_tickers}")

            normalized = df / df.iloc[0] * 100
            avg_group = normalized.mean(axis=1)
            group_df = pd.DataFrame({
                'Date': avg_group.index,
                'NormalizedPrice': avg_group.values,
                'Market': group_name
            })
            all_markets.append(group_df)
            time.sleep(5)
            plt.plot(avg_group.index, avg_group, label=group_name)
        except Exception as e:
            print(f"Error downloading or processing {group_name}: {e}")

    plt.title('Normalized Market Movements by Region/Type (Last Year)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price (Starting at 100)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    final_df = pd.concat(all_markets)
    final_df.to_csv('market_data.csv', index=False)



plot_movements()

