import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def get_data():
    df = pd.read_csv('macro_data_with_rolling.csv', parse_dates=['Date'])
    df.set_index('Date', inplace=True)

    start_date = df.index.min().strftime('%Y-%m-%d')
    end_date = df.index.max().strftime('%Y-%m-%d')

    target = yf.download('^GSPC', start=start_date, end=end_date, auto_adjust=True)
    if isinstance(target.columns, pd.MultiIndex):
        target.columns = target.columns.get_level_values(0)
    target = target[['Close']].rename(columns={'Close': 'S&P500'})

    combined = df.join(target, how='inner')
    combined = combined.ffill()
    combined['Future_Price'] = combined['S&P500'].shift(-10)
    combined.dropna(subset=['Future_Price'], inplace=True)
    threshold = 0.01  # 1% change
    change = ((combined["Future_Price"] - combined["S&P500"]) / combined["S&P500"])
    print(change.min())  # smallest negative change
    print(change[change < 0].describe())  # stats on all negative changes

    combined['Target'] = pd.cut(change, bins=[-np.inf, -0.01, 0.01, np.inf], labels=[0, 1, 2])
    combined['Target'] = combined['Target'].astype(int)
    features = combined.drop(columns=['S&P500','Future_Price', 'Target'])
    target = combined['Target']
    return features, target

def set_data():
    features, target = get_data()
    features.to_csv('features.csv', index=True)
    target.to_csv('target.csv', index=True)

def test_classifier():
    X, y = get_data()

    X = X.loc[:, X.nunique() > 1]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.ffill().bfill()

    # Optional: scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.95)  # keep 95% variance
    X_scaled = pca.fit_transform(X_scaled)

    # Split data (no shuffle to preserve time series order)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, shuffle=False)

    # Initialize classifier
    clf = RandomForestClassifier(random_state=42,class_weight='balanced', n_estimators=200, max_depth=10)
    
    print("Class distribution:\n", y.value_counts(normalize=True))

    # Train model
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    features_list = X.columns if hasattr(X, 'columns') else [f'feat_{i}' for i in range(X.shape[1])]
    print(features_list, importances)
    plt.figure(figsize=(10,6))
    plt.title("Feature importances")
    plt.bar(range(20), importances[indices[:20]], align="center")
    plt.xticks(range(20), [features_list[i] for i in indices[:20]], rotation=90)
    plt.tight_layout()
    plt.show()

'''set_data()
target = pd.read_csv('target.csv', parse_dates=['Date'])
target.set_index('Date', inplace=True)
features = pd.read_csv('features.csv', parse_dates=['Date'])
features.set_index('Date', inplace=True)

predict(features, target)'''


test_classifier()