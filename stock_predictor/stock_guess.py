import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta
import sentiment_finder as sf
import os
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
today = datetime.today()
minus_time = today - timedelta(days=180)

'''

Potential adds for getting info
* aplaca api - can do news research
* https://aletheiaapi.com/docs/ - has better historical and current data



'''

def get_sentiment_data(sentiment_dir="sentiment_data"):
    files = os.listdir(sentiment_dir)
    sentiment_files = [f for f in files if f.endswith('_sentiment.pkl')]
    #print("Sentiment files found:", sentiment_files)

    dfs = []

    for file in sentiment_files:
        ticker = file.split('_')[0]
        filepath = os.path.join(sentiment_dir, file)
        df = pd.read_pickle(filepath)

        if 'date' in df.columns:
            df = df.set_index('date')

        df.index = pd.to_datetime(df.index)
        u_ticker = ticker.upper()
        # Unpack list column into 3 columns
        if u_ticker in df.columns:
            sentiment_lists = df[u_ticker].tolist()
            sentiment_scores = [x[0] - x[1] for x in sentiment_lists]
            unpacked = pd.DataFrame(sentiment_scores, columns=[f'{u_ticker}_sentiment'], index=df.index)
            unpacked.index.name = 'date'
            unpacked = unpacked.groupby(pd.Grouper(freq='D')).mean()
            dfs.append(unpacked)
        else:
            print(f"Ticker {ticker} not found in columns: {df.columns}")

    return dfs


def load_and_process_sentiment(ticker):
    dfs = get_sentiment_data()
    for df in dfs:
        cols = [col for col in df.columns if col.startswith(ticker)]
        if cols:
            return df[cols]
    raise ValueError(f"No sentiment data found for ticker: {ticker}")

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
    data = data.bfill()

    '''try:
        sentiment_df = load_and_process_sentiment(ticker)
        sentiment_df.index = pd.to_datetime(sentiment_df.index).normalize()
        data['Date'] = pd.to_datetime(data['Date']).dt.normalize()
        data = data.merge(sentiment_df, left_on='Date', right_index=True, how='left')
        data = data.sort_values('Date')
        data = data.ffill()
    except FileNotFoundError as e:
        print(e)'''

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

    data = data.drop(columns=['High', 'Low', 'Open', 'dollar_volume'])
    data = data.bfill()


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
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_all, test_size=0.15, random_state=42, shuffle=False)
    model = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators': [300,400],
        'max_depth': [5, 7],
        'learning_rate': [0.01, 0.005],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1,3],
        'subsample': [0.6,0.8],
        'max_features': ['sqrt', 0.8, 'log2']
    }


    cv_splitter = KFold(n_splits=10, shuffle=True, random_state=42)
    
    grid = GridSearchCV(model, param_grid, cv= cv_splitter, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    # now need to match best hights

    y_pred = np.array(y_pred)
    np_y_test = np.array(y_test)
    #offset = np.mean(y_test - y_pred)
    y_pred_adjusted = y_pred #+ offset

    #adding the predicted price change of the last day to the price of the current day

    current_close = time_data['Close'].values[-len(np_y_test):]
    predicted_close = current_close + y_pred_adjusted

    mse = mean_squared_error(current_close, predicted_close)
    r2 = r2_score(current_close, predicted_close)
    print(f"\nBest params: {grid.best_params_}")
    print(f'\n\n mse: {mse}\nr2: {r2}\n\n')

    return model, y_test, predicted_close, preprocessor#, offset


def find_best_model_prediction(X_all, y_all, time_data):
    numeric_features = X_all.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), numeric_features)]
    )

    models = {
        "GradientBoostingRegressor": (
            GradientBoostingRegressor(random_state=42),
            {
                'n_estimators': [300,200],
                'max_depth': [5, 7],
                'learning_rate': [0.01, 0.005],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1,3],
                'subsample': [0.6,0.8],
                'max_features': ['sqrt', 0.8, 'log2']
            }
        ),


        "HistGradientBoosting": (
            HistGradientBoostingRegressor(random_state=42),
            {
                'max_iter': [300, 200],
                'max_depth': [5, 7],
                'learning_rate': [0.01, 0.005],
                'min_samples_leaf': [5,20, 30],
                'l2_regularization': [0.0, 0.1]
            }
        ),
        "RandomForest": (
            RandomForestRegressor(random_state=42),
            {
                'n_estimators': [200, 300],
                'max_depth': [5, 10],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 3],
                'max_features': ['sqrt', 'log2']
            }
        ),
        "MLPRegressor": (
            MLPRegressor(random_state=42, max_iter=5000),
            {
                'hidden_layer_sizes': [(50, 50),(50,), (100,), (100, 50)],
                'activation': ['relu', 'tanh'],
                'learning_rate_init': [0.001, 0.005, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'solver': ['adam'],
                'alpha': [0.0001, 0.001, 0.01]
            }
        ),
        "XGBoost": (
            XGBRegressor(random_state=42, verbosity=0),
            {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.001],
                'subsample': [0.6, 0.8],
                'reg_alpha': [0, 0.1],
                'min_child_weight': [3, 5],
                'gamma': [0.1, 0.3],
                'colsample_bytree': [0.8]
            }
        )
    }

    X_preprocessed = preprocessor.fit_transform(X_all)
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_all, test_size=0.15, random_state=42, shuffle=False)

    best_overall = None
    best_score = float('inf')
    best_model_name = None

    for name, (model, param_grid) in models.items():
        print(f"\nTuning {name}...")
        grid = GridSearchCV(model, param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train, y_train)

        y_pred = grid.best_estimator_.predict(X_test)
        y_pred_adjusted = y_pred  # You can add an offset correction here if needed

        current_close = time_data['Close'].values[-len(y_test):]
        predicted_close = current_close + y_pred_adjusted

        mse = mean_squared_error(current_close, predicted_close)
        r2 = r2_score(current_close, predicted_close)

        print(f"{name} best params: {grid.best_params_}")
        print(f"{name} MSE: {mse:.4f}, R2: {r2:.4f}")

        if mse < best_score:
            best_score = mse
            best_overall = grid.best_estimator_
            best_model_name = name
            best_y_test = y_test
            best_predicted_close = predicted_close

    print(f"\n✅ Best model: {best_model_name} with MSE: {best_score:.4f}")
    return best_overall, best_y_test, best_predicted_close, preprocessor


def plot_predictions(y_test, predicted_close, time_data):
    actual_close = time_data.loc[y_test.index, 'Close']
    predicted_series = pd.Series(predicted_close, index=y_test.index)
    predicted_series.index = predicted_series.index + 1  
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



def run_test(ticker, start_check = minus_time,end_check=today, risk_free_rate=0, lag_time='week'):
    data = get_metrics(ticker, start_check = start_check, end_check=end_check, risk_free_rate = risk_free_rate)
    X_all, y_all, time_data = create_params(data, lag_time=lag_time)
    model, y_test, predicted_close, preprocessor = find_best_model_prediction(X_all, y_all, time_data)
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


    '''try:
        sentiment_df = load_and_process_sentiment(ticker)
        sentiment_df.index = pd.to_datetime(sentiment_df.index).normalize()
        data['Date'] = pd.to_datetime(data['Date']).dt.normalize()
        data = data.merge(sentiment_df, left_on='Date', right_index=True, how='left')
        data = data.sort_values('Date')
        data = data.ffill()
    except FileNotFoundError as e:
        print(e)'''

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

    data = data.drop(columns=['High', 'Low', 'Open', 'dollar_volume'])
    data = data.fillna(method='bfill')

    data['returns_7d_mean'] = data['returns'].rolling(window=7).mean().bfill()
    data['volatility_7d'] = data['returns'].rolling(window=7).std().bfill()
    data = data.drop(columns=['Date'])

    print(data.iloc[[-1]])

    return data.iloc[[-1]], data['rolling_volatility'].iloc[-1], data['Close'].iloc[-1]

def next_prediction(X_new, model, preprocessor):
    X_new_preprocessed = preprocessor.transform(X_new)
    predicted_change = model.predict(X_new_preprocessed)[0]
    pred_adjusted = predicted_change
    last_close = X_new['Close'].values[0]
    final_guess = pred_adjusted + last_close
    print(f'predicted_adjusted= {pred_adjusted}')
    print(f'the next expected value is {final_guess}')
    return final_guess


def run_predict(ticker, start_check = minus_time,end_check=today, risk_free_rate=0, lag_time='week'):
    data = get_metrics(ticker, start_check = start_check, end_check=end_check, risk_free_rate = risk_free_rate)
    X_all, y_all, time_data = create_params(data, lag_time=lag_time)
    model, y_test, predicted_close, preprocessor = find_best_model_prediction(X_all, y_all, time_data)
    X_new = get_current_metrics(ticker)
    guess = next_prediction(X_new, model, preprocessor)
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
'''
tickers = ['AAPL', 'KO', 'MSFT', 'AMZN', 'NVDA', 'JPM', 'GOOG',"PLTR", "NFLX", "MCD", "SPY", "JNJ", "XOM", "META"]
X, features = cluster_volatility(tickers)
df = run_test('AAPL', start_check='2018-01-01', end_check='2025-06-01', risk_free_rate=0.02, lag_time='week')
plot_clusters(X,features)'''
tickers = ['AAPL', 'KO', 'MSFT', 'AMZN', 'NVDA', 'JPM', 'GOOG',"PLTR", "NFLX", "MCD", "SPY", "JNJ", "XOM", "META"]
def get_future_prices(tickers):
    predicted_prices = []
    for ticker in tickers:
        guess = run_predict(ticker, start_check='2020-01-01',end_check='2025-07-29', risk_free_rate=0.02, lag_time='week')
        print(f'{ticker}: {guess}')
        ticker_prices = {ticker : guess}
        predicted_prices.append(ticker_prices)
    
    return predicted_prices

def monte_carlo_with_model(model, start_price, X_base, preprocessor, n_simulations = 500, n_days =30, volatility = 0.02):
    dt = 1 / 252
    volatility = max(volatility, 0.02)
    feature_means = X_base.mean()
    feature_stds = X_base.std()

    simulations = np.zeros((n_simulations, n_days + 1))
    simulations[:, 0] = float(start_price)
    for sim in range(n_simulations):
        price = start_price
        current_features = np.random.normal(loc=feature_means, scale=feature_stds)
        for day in range(1, n_days + 1):
            noise = np.random.normal(loc=0.0, scale=feature_stds * 0.2)
            current_features = current_features + noise
            if 'Close' in X_base.columns:
                close_idx = X_base.columns.get_loc('Close')
                current_features[close_idx] = price
            random_features_df = pd.DataFrame([current_features], columns=X_base.columns)
            transformed_features = preprocessor.transform(random_features_df)
            predicted_change =  model.predict(transformed_features)[0]
            predicted_price = price + predicted_change
            predicted_return = np.log(predicted_price / price)
            # Add some randomness to simulate uncertainty
            drift = (predicted_return - 0.5 * volatility**2) * dt
            shock = volatility * np.sqrt(dt) * np.random.normal()
            log_price = drift + shock

            price *= np.exp(log_price)
            simulations[sim, day] = price

        #print(f'final price on sim {sim}: ${simulations[sim, -1]:.2f}')
    final_prices = simulations[:, -1]
    avg_price = np.mean(final_prices)
    max_price = np.max(final_prices)
    min_price = np.min(final_prices)
    pct_change_avg = (avg_price - start_price) / start_price * 100


    print(f"\nMonte Carlo Simulation Results after {n_days} days:")
    print(f"Average Final Price: ${avg_price:.2f}")
    print(f"Highest Final Price: ${max_price:.2f}")
    print(f"Lowest Final Price : ${min_price:.2f}")
    print(f"Average % Change: {pct_change_avg:.2f}%")
    return simulations

def plot_monte_carlo(simulations):
    plt.figure(figsize=(12,6))
    for i in range(100):  # plot only 100 simulations for clarity
        plt.plot(simulations[i], alpha=0.1, color='blue')

    plt.title("Monte Carlo Simulation of Predicted Stock Prices")
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()

def monte_carlo(ticker, start_check = '2020-01-01',end_check='2025-07-29', risk_free_rate=0.2, lag_time='week'):
    data = get_metrics(ticker, start_check = start_check, end_check=end_check, risk_free_rate = risk_free_rate)
    X_all, y_all, time_data = create_params(data, lag_time=lag_time)
    model, y_test, predicted_close, preprocessor = find_best_model_prediction(X_all, y_all, time_data)
    X_new , volatility, close = get_current_metrics(ticker)
    simulations  = monte_carlo_with_model(model, close, X_new, preprocessor, volatility=volatility)
    plot_monte_carlo(simulations)

monte_carlo('AAPL')
