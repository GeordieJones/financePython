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
from datetime import datetime, timedelta

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
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_all, test_size=0.2, random_state=42, shuffle=True)

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_pred = np.array(y_pred)
    np_y_test = np.array(y_test)
    offset = np.mean(y_test - y_pred)
    y_pred_adjusted = y_pred + offset

    mse = mean_squared_error(y_test, y_pred_adjusted)
    r2 = r2_score(y_test, y_pred_adjusted)

    print(f'\n\n mse: {mse}\nr2: {r2}\n\n')

    return y_pred_adjusted, y_test, model, X_test, preprocessor

def plot_predictions(y_test, predicted_close):
    actual_close = y_test.copy()
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

def run_predictions():
    df = pd.read_pickle('combined_data.pkl')
    X_all, y_all = preprocess_data(df)
    y_pred_adjusted, y_test, model, X_test, preprocessor = prediction(X_all, y_all)
    plot_predictions(y_test, y_pred_adjusted)

if __name__ == "__main__":
    run_predictions()
