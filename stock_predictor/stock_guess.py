import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from datetime import datetime, timedelta

'''

Potential adds for getting info
* aplaca api - can do news research
* https://aletheiaapi.com/docs/ - has better historical and current data



'''


def get_metrics(ticker, start_check = '2015-01-01',end_check='2025-06-01', risk_free_rate=0):
    data = yf.download(ticker, start=start_check, end=end_check)
    data.columns = data.columns.get_level_values(0)
    data.columns.name = None
    data = data.reset_index()
    data['garman_klass_vol'] = (((np.log(data['High']) - np.log(data['Low']))**2)/2) - ((2*np.log(2)-1)*(np.log(data['Close'])-np.log(data['Open']))**2)
    data['rsi'] = pandas_ta.rsi(close=data['Close'], length=14)

    bands = pandas_ta.bbands(close=(data['Close']), length=20)
    data['bb_low'] = bands['BBL_20_2.0']
    data['bb_mid'] = bands['BBM_20_2.0']
    data['bb_high'] = bands['BBU_20_2.0']

    atr = pandas_ta.atr(high=data['High'], low=data['Low'], close=data['Close'], length=14)
    data['atr'] = (atr - atr.mean()) / atr.std()

    macd = pandas_ta.macd(close=data['Close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
    data['macd'] = (macd - macd.mean()) / macd.std()

    data['dollar_volume'] = (data['Close']*data['Volume'])/1e6

    spy = yf.download('SPY', start='2015-01-01', end='2025-06-01')
    spy.columns = spy.columns.get_level_values(0)
    spy.columns.name = None

    spy = spy.reset_index()
    data['returns'] = data['Close'].pct_change()
    spy['returns'] = spy['Close'].pct_change()
    data['market returns'] = spy['returns']

    window = 30
    beta_values = [np.nan] * (window - 1)

    returns = pd.DataFrame({
        'stock': data['returns'],
        'market': spy['returns']
    }).dropna()

    for i in range(window - 1, len(returns)):
        window_data = returns.iloc[i - window + 1 : i + 1]
        cov = np.cov(window_data['stock'], window_data['market'])[0, 1]
        var = np.var(window_data['market'])
        beta = cov / var if var != 0 else np.nan
        beta_values.append(beta)

    beta_series = pd.Series(beta_values, index=returns.index)
    data.loc[beta_series.index, 'beta'] = beta_series

    data = data.bfill()
    data['alpha'] = data['returns'] - data['beta'] * data['market returns']
    rolling_mean = data['returns'].rolling(window).mean() - risk_free_rate
    rolling_std = data['returns'].rolling(window).std()
    data['rolling_sharpe'] = rolling_mean / rolling_std


    def downside_std(returns):
        negative_returns = returns[returns < 0]
        return negative_returns.std()

    data['downside_std'] = data['returns'].rolling(window).apply(downside_std, raw=False)
    data['rolling_sortino'] = rolling_mean / data['downside_std']

    data['rolling_volatility'] = data['returns'].rolling(window).std()

    data['momentum_20'] = data['Close'] / data['Close'].shift(20) - 1

    data['excess_return'] = data['returns'] - data['market returns']

    data['adx'] = pandas_ta.adx(high=data['High'], low=data['Low'], close=data['Close'], length=14)['ADX_14']

    data = data.drop(columns=['High', 'Low', 'Open', 'Volume','dollar_volume'])
    data = data.fillna(method='bfill')

    data['returns_7d_mean'] = data['returns'].rolling(window=7).mean().bfill()
    data['volatility_7d'] = data['returns'].rolling(window=7).std().bfill()
    data = data.drop(columns=['Date'])

    return data


def create_params(data, lag_time='week'):
    time_data = data.copy()

    match lag_time:
        case 'week':
            time_data = data.iloc[::5].reset_index(drop=True)
        case 'month':
            time_data = data.iloc[::28].reset_index(drop=True)

    time_data['target'] = (time_data['returns']*time_data['Close']).shift(-1)
    y_all = time_data['target'].dropna()
    X_all = time_data.loc[y_all.index].drop(columns=['target'])

    if len(X_all) == len(y_all):
        return X_all, y_all, time_data
    else:
        raise IndexError("Mismatch between X_all and y_all lengths")


def prediction(X_all, y_all, time_data):
    numeric_features = X_all.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ]
    )

    X_preprocessed = preprocessor.fit_transform(X_all)
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_all, test_size=0.2, random_state=42, shuffle=False)

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # now need to match best hights

    y_pred = np.array(y_pred)
    np_y_test = np.array(y_test)
    offset = np.mean(y_test - y_pred)
    y_pred_adjusted = y_pred + offset

    #adding the predicted price change of the last day to the price of the current day

    current_close = time_data['Close'].values[-len(np_y_test):]
    predicted_close = current_close + y_pred_adjusted

    mse = mean_squared_error(current_close, predicted_close)
    r2 = r2_score(current_close, predicted_close)
    print(f'\n\n mse: {mse}\nr2: {r2}\n\n')

    return model, y_test, predicted_close, offset, preprocessor

def plot_predictions(y_test, predicted_close, time_data):
    actual_close = time_data.loc[y_test.index, 'Close']
    predicted_series = pd.Series(predicted_close, index=y_test.index)
    mse = mean_squared_error(actual_close, predicted_series)
    r2 = r2_score(actual_close, predicted_series)

    print("MSE:", mse)
    print("R2:", r2)
    plt.figure(figsize=(10,6))
    plt.plot(actual_close, label='Actual Price', alpha=0.7)
    plt.plot(predicted_series, label='Predicted Price', alpha=0.7)
    plt.title('Gradient Boosting: Actual vs Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()



def run_test(ticker, start_check = '2015-01-01',end_check='2025-06-01', risk_free_rate=0, lag_time='week'):
    data = get_metrics(ticker, start_check = start_check, end_check=end_check, risk_free_rate = risk_free_rate)
    X_all, y_all, time_data = create_params(data, lag_time=lag_time)
    model, y_test, predicted_close, offset, preprocessor = prediction(X_all, y_all, time_data)
    plot_predictions(y_test, predicted_close, time_data)


def get_current_metrics(ticker, risk_free_rate = 0):
    today = datetime.today()
    one_month_ago = today - timedelta(days=60)
    data = yf.download(ticker, start=one_month_ago.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'), progress=False)

    data.columns = data.columns.get_level_values(0)
    data.columns.name = None
    data = data.reset_index()
    data['garman_klass_vol'] = (((np.log(data['High']) - np.log(data['Low']))**2)/2) - ((2*np.log(2)-1)*(np.log(data['Close'])-np.log(data['Open']))**2)
    data['rsi'] = pandas_ta.rsi(close=data['Close'], length=14)

    bands = pandas_ta.bbands(close=(data['Close']), length=20)
    data['bb_low'] = bands['BBL_20_2.0']
    data['bb_mid'] = bands['BBM_20_2.0']
    data['bb_high'] = bands['BBU_20_2.0']

    atr = pandas_ta.atr(high=data['High'], low=data['Low'], close=data['Close'], length=14)
    data['atr'] = (atr - atr.mean()) / atr.std()

    macd = pandas_ta.macd(close=data['Close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
    data['macd'] = (macd - macd.mean()) / macd.std()

    data['dollar_volume'] = (data['Close']*data['Volume'])/1e6

    spy = yf.download('SPY', start='2015-01-01', end='2025-06-01')
    spy.columns = spy.columns.get_level_values(0)
    spy.columns.name = None

    spy = spy.reset_index()
    data['returns'] = data['Close'].pct_change()
    spy['returns'] = spy['Close'].pct_change()
    data['market returns'] = spy['returns']

    window = 30
    beta_values = [np.nan] * (window - 1)

    returns = pd.DataFrame({
        'stock': data['returns'],
        'market': spy['returns']
    }).dropna()

    for i in range(window - 1, len(returns)):
        window_data = returns.iloc[i - window + 1 : i + 1]
        cov = np.cov(window_data['stock'], window_data['market'])[0, 1]
        var = np.var(window_data['market'])
        beta = cov / var if var != 0 else np.nan
        beta_values.append(beta)

    beta_series = pd.Series(beta_values, index=returns.index)
    data.loc[beta_series.index, 'beta'] = beta_series

    data = data.bfill()
    data['alpha'] = data['returns'] - data['beta'] * data['market returns']
    rolling_mean = data['returns'].rolling(window).mean() - risk_free_rate
    rolling_std = data['returns'].rolling(window).std()
    data['rolling_sharpe'] = rolling_mean / rolling_std


    def downside_std(returns):
        negative_returns = returns[returns < 0]
        return negative_returns.std()

    data['downside_std'] = data['returns'].rolling(window).apply(downside_std, raw=False)
    data['rolling_sortino'] = rolling_mean / data['downside_std']

    data['rolling_volatility'] = data['returns'].rolling(window).std()

    data['momentum_20'] = data['Close'] / data['Close'].shift(20) - 1

    data['excess_return'] = data['returns'] - data['market returns']

    data['adx'] = pandas_ta.adx(high=data['High'], low=data['Low'], close=data['Close'], length=14)['ADX_14']

    data = data.drop(columns=['High', 'Low', 'Open', 'Volume','dollar_volume'])
    data = data.fillna(method='bfill')

    data['returns_7d_mean'] = data['returns'].rolling(window=7).mean().bfill()
    data['volatility_7d'] = data['returns'].rolling(window=7).std().bfill()
    data = data.drop(columns=['Date'])

    print(data.iloc[[-1]])

    return data.iloc[[-1]]

def next_prediction(X_new, model, offset, preprocessor):
    X_new_preprocessed = preprocessor.transform(X_new)
    predicted_change = model.predict(X_new_preprocessed)[0]
    pred_adjusted = predicted_change + offset
    last_close = X_new['Close'].values[0]
    final_guess = pred_adjusted + last_close
    print(f'predicted_change= {predicted_change}')
    print(f'offset {offset}')
    print(f'predicted_adjusted= {pred_adjusted}')
    print(f'the next expected value is {final_guess}')
    return final_guess


def run_predict(ticker, start_check = '2015-01-01',end_check='2025-06-01', risk_free_rate=0, lag_time='week'):
    data = get_metrics(ticker, start_check = start_check, end_check=end_check, risk_free_rate = risk_free_rate)
    X_all, y_all, time_data = create_params(data, lag_time=lag_time)
    model, y_test, predicted_close, offset, preprocessor = prediction(X_all, y_all, time_data)
    X_new = get_current_metrics(ticker)
    guess = next_prediction(X_new, model, offset, preprocessor)
    return guess


def cluster_volatility(tickers, start ='2025-01-01', end = '2025-06-01', n_clusters=3):
    price_data = yf.download(tickers, start=start, end=end,auto_adjust=False, progress=False)['Close']
    returns = price_data.pct_change().dropna()
    features = pd.DataFrame(index=tickers)
    features['mean_return'] = returns.mean()
    features['volatility'] = returns.std()
    features['sharpe'] = features['mean_return'] / features['volatility']

    def max_drawdown(series):
        cum_returns = (1 + series).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        return drawdown.min()
    
    features['max_drawdown'] = returns.apply(max_drawdown)

    features = features.dropna()
    X = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters,n_init=10, random_state=42)
    features['cluster'] = kmeans.fit_predict(X)

    return X, features


def plot_clusters(X, features):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=features['cluster'], cmap='Set1', s=100, alpha=0.8)
    for i, ticker in enumerate(features.index):
        plt.text(X_pca[i, 0] + 0.02, X_pca[i, 1], ticker, fontsize=9)

    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('Stock Clustering by Risk/Reward Profile')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True)
    plt.show()

#testing clustering

tickers = ['AAPL', 'KO', 'MSFT', 'AMZN', 'NVDA', 'JPM', 'GOOG',"PLTR", "NFLX", "ARKK", "SOFI", "MCD", "PG", "JNJ", "XOM", "BRK-B"]
X, features = cluster_volatility(tickers)
plot_clusters(X,features)
