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
    final_gain = []
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
            final_gain.append(group_df['NormalizedPrice'].iloc[-1])
            print(f"\nfinal return for {group_df['Market'].iloc[-1]}: {group_df['NormalizedPrice'].iloc[-1]}")
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
    group_names = list(market_groups.keys())
    avg_ret = sum(final_gain)/ len(final_gain)
    best_return = 0
    best_ticker = ''
    for i in range(len(final_gain)):
        if final_gain[i] > best_return:
            best_return = final_gain[i]
            best_ticker = group_names[i]

    print(f'the best return was {best_ticker} with a return of {best_return}\n average return was {avg_ret}')



plot_movements()

