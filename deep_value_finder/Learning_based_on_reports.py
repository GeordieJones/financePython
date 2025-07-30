import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, KFold
from datetime import datetime, timedelta
from sklearn.metrics import make_scorer, mean_pinball_loss
import joblib
from tqdm import tqdm
import yfinance as yf


def all_data_in_one():

    # sets up and cleans into a single dataframe

    df = pd.DataFrame()
    for year_i in range(4):
        year_df = pd.read_pickle(f'year{year_i}_data_labeled.pkl')
        df = pd.concat([df, year_df], ignore_index=True)

    threshold = .9  # Minimum % of non-NaN values to keep a column
    min_non_na = int(threshold * len(df))
    df = df.dropna(axis=1, thresh=min_non_na)
    df = df.dropna()

    df.to_pickle('combined_data.pkl') 

    return df

def preprocess_data(df):
    data = df.copy()
    data = data.sort_values(by='Close_9mo_after') # makes the graph at the end look better
    y_all = data['Close_9mo_after']
    data.drop(columns=['Close_9mo_after'], inplace=True)

    return data, y_all

def prediction(X_all, y_all):
    numeric_features = X_all.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ]
    )

    X_preprocessed = preprocessor.fit_transform(X_all)
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_all, test_size=0.15, random_state=42, shuffle=True)

    model = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators': [400],
        'max_depth': [9],
        'learning_rate': [0.02],
        'min_samples_split': [2],
        'min_samples_leaf': [1,3],
        'subsample': [0.6,0.8],
        'max_features': ['sqrt', 0.8]
    }

    cv_splitter = KFold(n_splits=10, shuffle=True, random_state=42)
    
    grid = GridSearchCV(model, param_grid, cv= cv_splitter, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    offset = np.mean(y_test - y_pred)
    y_pred_adjusted = y_pred + offset
    
    mse = mean_squared_error(y_test, y_pred_adjusted)
    r2 = r2_score(y_test, y_pred_adjusted)
    
    print(f"Best params: {grid.best_params_}")
    print(f"MSE: {mse}")
    print(f"R2: {r2}")
    
    return y_pred, y_test, best_model, X_test, preprocessor

def plot_predictions(y_test, predicted_close):
    actual_close = y_test.copy()
    predicted_series = pd.Series(predicted_close).reset_index(drop=True)
    actual_close = y_test.reset_index(drop=True)
    mse = mean_squared_error(actual_close, predicted_series)
    r2 = r2_score(actual_close, predicted_series)

    print("MSE:", mse)
    print("R2:", r2)
    
    actual = y_test.reset_index(drop=True)
    predicted = pd.Series(predicted_close).reset_index(drop=True)
    stocks = np.arange(len(actual))  # just indices for stocks

    plt.figure(figsize=(12,6))
    plt.plot(stocks, actual, 'o-', label='Actual Price', alpha=0.7)
    plt.plot(stocks, predicted, 'x-', label='Predicted Price', alpha=0.7)
    plt.xlabel('Stock Number (sorted by actual price)')
    plt.ylabel('Price')
    plt.title('Predicted vs Actual Stock Prices (per stock index)')
    plt.legend()
    plt.show()

def run_learning():
    df = pd.read_pickle('combined_data.pkl')
    X_all, y_all = preprocess_data(df)
    y_pred_adjusted, y_test, model, X_test, preprocessor = prediction(X_all, y_all)
    plot_predictions(y_test, y_pred_adjusted)
    save_model(model, preprocessor)

def save_model(model, preprocessor, model_path='best_model.joblib', preproc_path='preprocessor.joblib'):
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preproc_path)
    print("Model and preprocessor saved!")

def load_model(model_path='best_model.joblib', preproc_path='preprocessor.joblib'):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preproc_path)
    print("Model and preprocessor loaded!")
    return model, preprocessor

def get_current_metrics():
    
    latest_known = datetime(2024, 9, 30)

    # Load tickers from your DataFrame
    df_ticker = pd.read_pickle('year0_data.pkl')
    tickers = df_ticker.index.tolist()  # Use .index, not 'symbol', since 'symbol' is the index in your dataset

    target_columns = [
    'Tax Rate For Calcs', 'Normalized EBITDA', 'EBITDA', 'EBIT',
    'Normalized Income', 'Total Expenses', 'Diluted EPS', 'Net Income',
    'Tax Provision', 'Pretax Income', 'Operating Income',
    'Operating Expense', 'Selling General And Administration',
    'Total Revenue'
]

    all_data = []
    date_for_ticker = {}

    for ticker in tqdm(tickers, desc="Fetching quarterly data"):
        try:
            q_fin = yf.Ticker(ticker).quarterly_financials
            if q_fin is None or q_fin.empty:
                continue

            latest_quarter = q_fin.columns[0]
            if latest_quarter <= latest_known:
                continue

            df = q_fin.T  # Transpose so each row is a quarter

            # Keep only desired columns that are present
            available_cols = [col for col in target_columns if col in df.columns]
            df = df[available_cols]

            # Drop rows with any NaNs
            df_clean = df.dropna()
            if df_clean.empty:
                continue

            # Take the most recent cleaned row
            most_recent = df_clean.iloc[0]
            most_recent.name = ticker  # Set index as ticker
            all_data.append(most_recent)
            date_for_ticker[ticker] = df_clean.index[0] 


        except Exception as e:
            print(f"Error with {ticker}: {e}")

    # Step 5: Create final DataFrame
    final_df = pd.DataFrame(all_data)

    # Optional: reset index if needed
    # final_df.reset_index(inplace=True)

    # Step 6: Preview
    print(f"\nFinal DataFrame with shape: {final_df.shape}")
    print(final_df.head())
    return final_df, date_for_ticker

if __name__ == "__main__":
    model,preprocessor = load_model()
    print('Loaded model and preprocessor')
    