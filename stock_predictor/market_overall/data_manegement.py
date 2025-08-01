import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import time
from fredapi import Fred

API_KEY = 'ba62bc0647e66e6897498b414e0eb249'
fred = Fred(api_key=API_KEY)

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


def macro_data():
    series = {
        'Inflation_CPI': 'CPIAUCSL',
        'Fed_Funds_Rate': 'FEDFUNDS',
        'Unemployment_Rate': 'UNRATE',
        'USD_Index': 'DTWEXBGS',
        'VIX fear': 'VIXCLS',
        'Yield_Curve_10Y_2Y': 'T10Y2Y',
        'Consumer_Confidence': 'UMCSENT',
        'Housing_Starts': 'HOUST',
        'M2 money supply': 'M2SL',
        'reserve repo': 'RRPONTSYD',
        'Leading_Indicators': 'USSLIND',
        'USA': 'GDP',
        'bank reserves': 'WRESBAL',
        'China': 'CHNGDPNQDSMEI',
        'India': 'INDGDPNQDSMEI',
        'Germany': 'DEUGDPNQDSMEI',
        'Japan': 'JPNNGDP',
        'SOFR': 'SOFR',
        'retail sales': 'RSAFS'
    }
    macro_df = pd.DataFrame()
    for label, code in series.items():
        data = fred.get_series(code,observation_start='2006-01-01')
        data.index = pd.to_datetime(data.index)
        data.name = label
        macro_df = pd.concat([macro_df, data], axis=1)
        time.sleep(1)
    macro_df.index = pd.to_datetime(macro_df.index)
    macro_df = macro_df.resample('MS').ffill()
    return macro_df



series_dict = {
    'daily': {
        'VIX fear': 'VIXCLS',
        'SOFR': 'SOFR',
        'Fed_Funds_Rate': 'FEDFUNDS',
        '10Y Treasury': 'GS10',
        'BAA Spread': 'BAA10Y',
        'S&P 500': 'SP500'
    },
    'weekly': {
        'USD_Index': 'DTWEXBGS',
        'reserve repo': 'RRPONTSYD',
        'bank reserves': 'WRESBAL',
        'Initial Jobless Claims': 'ICSA'
    },
    'monthly': {
        'Inflation_CPI': 'CPIAUCSL',
        'Unemployment_Rate': 'UNRATE',
        'Consumer_Confidence': 'UMCSENT',
        'Retail Sales': 'RSAFS',
        'Yield_Curve_10Y_2Y': 'T10Y2Y',
        'M2 Money Supply': 'M2SL',
        'Industrial Production': 'INDPRO',
        'ISM Manufacturing PMI': 'NAPM',
        'Real Personal Income': 'W875RX1',
        'Vehicle Sales': 'TOTALSA',
        'Capacity Utilization': 'TCU',
        'Chicago Fed Activity Index': 'CFNAI'
    },
    'quarterly': {
        'USA_GDP': 'GDP',
        'China_GDP': 'CHNGDPNQDSMEI',
        'India_GDP': 'INDGDPNQDSMEI',
        'Germany_GDP': 'DEUGDPNQDSMEI',
        'Japan_GDP': 'JPNNGDP',
        'UK_GDP': 'GBRNGDP'
    }
}

def fetch_series(group_name, series_map, freq):
    df = pd.DataFrame()
    for label, code in series_map.items():
        try:
            print(f"Fetching {label}...")
            data = fred.get_series(code, observation_start='2006-01-01')
            data = pd.DataFrame(data, columns=[label])
            data.index = pd.to_datetime(data.index)
            df = pd.concat([df, data], axis=1)
            time.sleep(1)  # Avoid hitting FRED rate limit
        except Exception as e:
            print(f"Failed to fetch {label}: {e}")
    return df.resample(freq).ffill()

def macro_data_combined(resample_to='D'):
    df_daily = fetch_series('daily', series_dict['daily'], 'D')
    df_weekly = fetch_series('weekly', series_dict['weekly'], 'W-FRI')
    df_monthly = fetch_series('monthly', series_dict['monthly'], 'MS')
    df_quarterly = fetch_series('quarterly', series_dict['quarterly'], 'QS')

    # Merge all DataFrames with outer join
    all_df = df_daily.join(df_weekly, how='outer')
    all_df = all_df.join(df_monthly, how='outer')
    all_df = all_df.join(df_quarterly, how='outer')

    # Resample to final unified frequency (default daily)
    all_df = all_df.resample(resample_to).ffill()

    return all_df

# Example usage
macro_df = macro_data_combined(resample_to='D')
macro_df.to_csv('macro_data.csv', index_label='Date')
print(macro_df.tail())
