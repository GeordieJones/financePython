import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

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
    y_test = np.array(y_test)
    offset = np.mean(y_test - y_pred)
    y_pred_adjusted = y_pred + offset

    #adding the predicted price change of the last day to the price of the current day

    current_close = time_data['Close'].values[-len(y_test):]
    predicted_close = current_close + y_pred_adjusted

    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))
    
    return model, y_test, predicted_close



print(get_metrics(ticker='AAPL'))