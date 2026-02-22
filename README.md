# -*- coding: utf-8 -*-
"""
Price Forecasting of Crude Oil Using Hybrid Machine Learning Models
and Implications for Financial Risk Management (2020-2025)
================================================================================
Ce script implémente plusieurs modèles hybrides pour la prévision du prix du pétrole brut WTI.
Il intègre :
- Lissage exponentiel adaptatif pour la période COVID-19
- Analyse exploratoire (prix, rendements, distribution)
- Modèles statistiques : ARIMA, SARIMA
- Modèles ML : LSTM, GRU, XGBoost, LightGBM
- Modèles hybrides : Prophet, Stacking
- Validation croisée temporelle (3 folds)
- Sélection du meilleur modèle
- Prévision future à 30 jours avec le meilleur modèle (lorsque possible)
- Calcul de métriques de risque (VaR, volatilité) pour la gestion financière
- Rapport HTML interactif avec tous les graphiques, tableaux et sections de risk management
- Archive ZIP des résultats
- Publication sur GitHub (optionnelle)
"""

# Installation des dépendances
!pip install pandas numpy matplotlib seaborn statsmodels scikit-learn tensorflow arch plotly jinja2 joblib yfinance xgboost lightgbm prophet pmdarima requests --quiet

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Input,
                                     Bidirectional, Conv1D, MaxPooling1D,
                                     GlobalAveragePooling1D, Add, LayerNormalization,
                                     MultiHeadAttention, Flatten, Concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import jinja2
import os
import zipfile
from scipy import stats
import subprocess
import getpass
import json

# Tentative d'import de requests pour l'API GitHub
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️ requests non installé, la création automatique du dépôt GitHub ne sera pas possible.")

# Configuration
os.makedirs('results_oil_price', exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
pd.set_option('display.float_format', '{:.4f}'.format)

# Paramètres
TICKER = 'CL=F'
START_DATE = '2020-01-01'
END_DATE = '2025-12-31'
TRAIN_CV_RATIO = 0.8
N_SPLITS = 3
FUTURE_DAYS = 30
WINDOW = 20
COVID_START = '2020-03-01'
COVID_END = '2020-06-30'
ALPHA_NORMAL = 0.3
ALPHA_COVID = 0.7
SEQUENCE_LENGTH = 30
EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 5
RANDOM_SEED = 42
VAR_CONFIDENCE = 0.95
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# 1. Téléchargement des données
print("📥 Téléchargement des données WTI...")
df = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
data_raw = df[['Close']].copy().dropna()
data_raw.columns = ['Price']
print(f"✅ Données chargées : {data_raw.shape[0]} observations, de {data_raw.index[0].date()} à {data_raw.index[-1].date()}")

# 2. Lissage exponentiel adaptatif
def adaptive_smoothing(series, alpha_norm, alpha_covid, covid_start, covid_end):
    smoothed = series.copy()
    ema = series.ewm(alpha=alpha_norm, adjust=False).mean()
    mask = (series.index >= pd.Timestamp(covid_start)) & (series.index <= pd.Timestamp(covid_end))
    if mask.any():
        ema_covid = series[mask].ewm(alpha=alpha_covid, adjust=False).mean()
        ema.loc[mask] = ema_covid
    return ema.fillna(method='ffill').fillna(method='bfill')

data = data_raw.copy()
data['Price_smoothed'] = adaptive_smoothing(data['Price'], ALPHA_NORMAL, ALPHA_COVID,
                                             COVID_START, COVID_END)

# Graphique 1 : original vs lissé
plt.figure(figsize=(12,5))
plt.plot(data.index, data['Price'], 'b-', alpha=0.4, label='Original')
plt.plot(data.index, data['Price_smoothed'], 'r-', label='Lissé (adaptatif)')
plt.axvspan(pd.Timestamp(COVID_START), pd.Timestamp(COVID_END), alpha=0.2, color='red', label='COVID-19')
plt.title('Prix WTI : original vs lissage adaptatif')
plt.xlabel('Date'); plt.ylabel('Prix (USD)')
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig('results_oil_price/price_smoothed.png', dpi=150)
plt.show()

# On utilise la série lissée
data['Price'] = data['Price_smoothed']
data = data[['Price']].copy()

# 3. Calcul des rendements et statistiques
data['returns'] = data['Price'].pct_change() * 100
data = data.dropna(subset=['returns']).copy()

# Graphique 2 : rendements
plt.figure(figsize=(12,5))
plt.plot(data.index, data['returns'], color='purple', linewidth=0.8)
plt.axvspan(pd.Timestamp(COVID_START), pd.Timestamp(COVID_END), alpha=0.2, color='red', label='COVID-19')
plt.axhline(y=0, color='black', linewidth=0.5)
plt.title('Rendements journaliers WTI (série lissée)')
plt.xlabel('Date'); plt.ylabel('Rendement (%)')
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig('results_oil_price/returns.png', dpi=150)
plt.show()

# Graphique 3 : distribution des rendements
fig, axes = plt.subplots(1,2, figsize=(14,5))
axes[0].hist(data['returns'], bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
x = np.linspace(data['returns'].min(), data['returns'].max(), 200)
axes[0].plot(x, stats.norm.pdf(x, data['returns'].mean(), data['returns'].std()), 'r-', label='Normale')
axes[0].set_title('Histogramme des rendements')
axes[0].set_xlabel('Rendement (%)')
axes[0].set_ylabel('Densité')
axes[0].legend()
stats.probplot(data['returns'], dist="norm", plot=axes[1])
axes[1].set_title('Q-Q plot')
plt.suptitle('Distribution des rendements WTI')
plt.tight_layout()
plt.savefig('results_oil_price/returns_distribution.png', dpi=150)
plt.show()

# 4. Création de features
def create_features(df, lags=5, windows=[5,10,20]):
    dataf = df.copy()
    for lag in range(1, lags+1):
        dataf[f'lag_{lag}'] = dataf['Price'].shift(lag)
    for w in windows:
        dataf[f'ma_{w}'] = dataf['Price'].rolling(window=w).mean()
        dataf[f'std_{w}'] = dataf['Price'].rolling(window=w).std()
    dataf['return'] = dataf['Price'].pct_change()
    return dataf.dropna()

data_feat = create_features(data)
print(f"✅ Features créées : {data_feat.shape[1]-1} variables explicatives")

# 5. Split train / test
split_idx = int(len(data_feat) * TRAIN_CV_RATIO)
train = data_feat.iloc[:split_idx].copy()
test = data_feat.iloc[split_idx:].copy()
print(f"\n📅 Train : {train.index[0].date()} → {train.index[-1].date()} ({len(train)} obs)")
print(f"📅 Test  : {test.index[0].date()} → {test.index[-1].date()} ({len(test)} obs)")

y_train = train['Price']
y_test = test['Price']
X_train = train.drop('Price', axis=1)
X_test = test.drop('Price', axis=1)

# Normalisation pour modèles non séquentiels
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1,1)).flatten()

# Préparation des séquences pour LSTM/GRU
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

price_series = data['Price'].loc[data_feat.index]
X_seq, y_seq = create_sequences(price_series.values, SEQUENCE_LENGTH)
seq_dates = price_series.index[SEQUENCE_LENGTH:]

train_seq_idx = int(len(X_seq) * TRAIN_CV_RATIO)
X_seq_train, X_seq_test = X_seq[:train_seq_idx], X_seq[train_seq_idx:]
y_seq_train, y_seq_test = y_seq[:train_seq_idx], y_seq[train_seq_idx:]
dates_seq_train = seq_dates[:train_seq_idx]
dates_seq_test = seq_dates[train_seq_idx:]

scaler_seq = StandardScaler()
X_seq_train_2d = X_seq_train.reshape(-1, SEQUENCE_LENGTH)
X_seq_test_2d = X_seq_test.reshape(-1, SEQUENCE_LENGTH)
X_seq_train_scaled = scaler_seq.fit_transform(X_seq_train_2d).reshape(-1, SEQUENCE_LENGTH, 1)
X_seq_test_scaled = scaler_seq.transform(X_seq_test_2d).reshape(-1, SEQUENCE_LENGTH, 1)

scaler_seq_y = StandardScaler()
y_seq_train_scaled = scaler_seq_y.fit_transform(y_seq_train.reshape(-1,1)).flatten()
y_seq_test_scaled = scaler_seq_y.transform(y_seq_test.reshape(-1,1)).flatten()

# 6. Fonctions des modèles
def train_arima(y, horizon):
    model = auto_arima(y, seasonal=False, trace=False, error_action='ignore', suppress_warnings=True)
    return model

def train_sarima(y, horizon):
    model = auto_arima(y, seasonal=True, m=30, trace=False, error_action='ignore', suppress_warnings=True)
    return model

def build_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True), Dropout(0.2),
        LSTM(32), Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model

def build_gru(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=True), Dropout(0.2),
        GRU(32), Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model

def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train):
    model = lgb.LGBMRegressor(objective='regression', n_estimators=100, max_depth=5, random_state=RANDOM_SEED, verbose=-1)
    model.fit(X_train, y_train)
    return model

def train_prophet(df):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    return model

def train_stacking(X_train, y_train, X_val, y_val):
    model1 = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=RANDOM_SEED)
    model2 = lgb.LGBMRegressor(n_estimators=50, max_depth=3, random_state=RANDOM_SEED, verbose=-1)
    model3 = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(32, activation='relu'), Dropout(0.2),
        Dense(16, activation='relu'), Dropout(0.2),
        Dense(1)
    ])
    model3.compile(optimizer=Adam(0.001), loss='mse')
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    model3.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    pred1 = model1.predict(X_val)
    pred2 = model2.predict(X_val)
    pred3 = model3.predict(X_val).flatten()
    meta_X = np.column_stack([pred1, pred2, pred3])
    from sklearn.linear_model import LinearRegression
    meta_model = LinearRegression()
    meta_model.fit(meta_X, y_val)
    return {'models': [model1, model2, model3], 'meta': meta_model}

# 7. Validation croisée temporelle
tscv = TimeSeriesSplit(n_splits=N_SPLITS)
model_names = ['ARIMA', 'SARIMA', 'LSTM', 'GRU', 'XGBoost', 'LightGBM', 'Prophet', 'Stacking']
cv_scores = {name: [] for name in model_names}
fold = 0

for train_idx, val_idx in tscv.split(train):
    fold += 1
    print(f"\n--- Fold {fold}/{N_SPLITS} ---")
    train_fold = train.iloc[train_idx]
    val_fold = train.iloc[val_idx]

    y_train_fold = train_fold['Price']
    y_val_fold = val_fold['Price']
    X_train_fold = train_fold.drop('Price', axis=1)
    X_val_fold = val_fold.drop('Price', axis=1)

    scaler_fold_X = StandardScaler()
    scaler_fold_y = StandardScaler()
    X_tr_scaled = scaler_fold_X.fit_transform(X_train_fold)
    X_val_scaled = scaler_fold_X.transform(X_val_fold)
    y_tr_scaled = scaler_fold_y.fit_transform(y_train_fold.values.reshape(-1,1)).flatten()
    y_val_scaled = scaler_fold_y.transform(y_val_fold.values.reshape(-1,1)).flatten()

    # Alignement des séquences pour LSTM/GRU
    train_fold_dates = train_fold.index
    val_fold_dates = val_fold.index
    date_to_seq_idx_train = {d: i for i, d in enumerate(dates_seq_train)}
    date_to_seq_idx_val = {d: i for i, d in enumerate(dates_seq_test)}
    train_seq_idx_fold = [date_to_seq_idx_train[d] for d in train_fold_dates if d in date_to_seq_idx_train]
    val_seq_idx_fold = [date_to_seq_idx_val[d] for d in val_fold_dates if d in date_to_seq_idx_val]

    if len(train_seq_idx_fold) > 0 and len(val_seq_idx_fold) > 0:
        X_tr_seq = X_seq_train_scaled[train_seq_idx_fold]
        y_tr_seq = y_seq_train_scaled[train_seq_idx_fold]
        X_val_seq = X_seq_test_scaled[val_seq_idx_fold]
        y_val_seq = y_seq_test_scaled[val_seq_idx_fold]
    else:
        X_tr_seq = np.zeros((1, SEQUENCE_LENGTH, 1))
        y_tr_seq = np.zeros(1)
        X_val_seq = np.zeros((1, SEQUENCE_LENGTH, 1))
        y_val_seq = np.zeros(1)

    # ARIMA
    arima_model = train_arima(y_train_fold, len(y_val_fold))
    pred_arima = arima_model.predict(n_periods=len(y_val_fold))
    rmse_arima = np.sqrt(mean_squared_error(y_val_fold, pred_arima))
    cv_scores['ARIMA'].append(rmse_arima)
    print(f"   ARIMA RMSE: {rmse_arima:.4f}")

    # SARIMA
    sarima_model = train_sarima(y_train_fold, len(y_val_fold))
    pred_sarima = sarima_model.predict(n_periods=len(y_val_fold))
    rmse_sarima = np.sqrt(mean_squared_error(y_val_fold, pred_sarima))
    cv_scores['SARIMA'].append(rmse_sarima)
    print(f"   SARIMA RMSE: {rmse_sarima:.4f}")

    # LSTM
    if len(X_tr_seq) > 1:
        lstm_model = build_lstm((SEQUENCE_LENGTH, 1))
        lstm_model.fit(X_tr_seq, y_tr_seq, epochs=20, batch_size=16, verbose=0)
        pred_lstm_scaled = lstm_model.predict(X_val_seq).flatten()
        pred_lstm = scaler_seq_y.inverse_transform(pred_lstm_scaled.reshape(-1,1)).flatten()
        pred_lstm_series = pd.Series(pred_lstm, index=[d for d in val_fold_dates if d in date_to_seq_idx_val])
        common_dates = pred_lstm_series.index.intersection(y_val_fold.index)
        rmse_lstm = np.sqrt(mean_squared_error(y_val_fold.loc[common_dates], pred_lstm_series.loc[common_dates]))
    else:
        rmse_lstm = np.nan
    cv_scores['LSTM'].append(rmse_lstm)
    print(f"   LSTM RMSE: {rmse_lstm:.4f}")

    # GRU
    if len(X_tr_seq) > 1:
        gru_model = build_gru((SEQUENCE_LENGTH, 1))
        gru_model.fit(X_tr_seq, y_tr_seq, epochs=20, batch_size=16, verbose=0)
        pred_gru_scaled = gru_model.predict(X_val_seq).flatten()
        pred_gru = scaler_seq_y.inverse_transform(pred_gru_scaled.reshape(-1,1)).flatten()
        pred_gru_series = pd.Series(pred_gru, index=[d for d in val_fold_dates if d in date_to_seq_idx_val])
        common_dates = pred_gru_series.index.intersection(y_val_fold.index)
        rmse_gru = np.sqrt(mean_squared_error(y_val_fold.loc[common_dates], pred_gru_series.loc[common_dates]))
    else:
        rmse_gru = np.nan
    cv_scores['GRU'].append(rmse_gru)
    print(f"   GRU RMSE: {rmse_gru:.4f}")

    # XGBoost
    xgb_model = train_xgboost(X_tr_scaled, y_tr_scaled)
    pred_xgb_scaled = xgb_model.predict(X_val_scaled)
    pred_xgb = scaler_fold_y.inverse_transform(pred_xgb_scaled.reshape(-1,1)).flatten()
    rmse_xgb = np.sqrt(mean_squared_error(y_val_fold, pred_xgb))
    cv_scores['XGBoost'].append(rmse_xgb)
    print(f"   XGBoost RMSE: {rmse_xgb:.4f}")

    # LightGBM
    lgb_model = train_lightgbm(X_tr_scaled, y_tr_scaled)
    pred_lgb_scaled = lgb_model.predict(X_val_scaled)
    pred_lgb = scaler_fold_y.inverse_transform(pred_lgb_scaled.reshape(-1,1)).flatten()
    rmse_lgb = np.sqrt(mean_squared_error(y_val_fold, pred_lgb))
    cv_scores['LightGBM'].append(rmse_lgb)
    print(f"   LightGBM RMSE: {rmse_lgb:.4f}")

    # Prophet
    prophet_train = train_fold[['Price']].reset_index()
    prophet_train.columns = ['ds', 'y']
    prophet_model = train_prophet(prophet_train)
    future = prophet_model.make_future_dataframe(periods=len(y_val_fold))
    forecast = prophet_model.predict(future)
    pred_prophet = forecast['yhat'].values[-len(y_val_fold):]
    rmse_prophet = np.sqrt(mean_squared_error(y_val_fold, pred_prophet))
    cv_scores['Prophet'].append(rmse_prophet)
    print(f"   Prophet RMSE: {rmse_prophet:.4f}")

    # Stacking
    split2 = int(0.8 * len(X_tr_scaled))
    if split2 == 0: split2 = 1
    X_tr2, y_tr2 = X_tr_scaled[:split2], y_tr_scaled[:split2]
    X_val2, y_val2 = X_tr_scaled[split2:], y_tr_scaled[split2:]
    if len(X_tr2) > 0 and len(X_val2) > 0:
        stacking_models = train_stacking(X_tr2, y_tr2, X_val2, y_val2)
        pred1 = stacking_models['models'][0].predict(X_val_scaled)
        pred2 = stacking_models['models'][1].predict(X_val_scaled)
        pred3 = stacking_models['models'][2].predict(X_val_scaled).flatten()
        meta_X = np.column_stack([pred1, pred2, pred3])
        pred_stacking_scaled = stacking_models['meta'].predict(meta_X)
        pred_stacking = scaler_fold_y.inverse_transform(pred_stacking_scaled.reshape(-1,1)).flatten()
        rmse_stacking = np.sqrt(mean_squared_error(y_val_fold, pred_stacking))
    else:
        rmse_stacking = np.nan
    cv_scores['Stacking'].append(rmse_stacking)
    print(f"   Stacking RMSE: {rmse_stacking:.4f}")

# 8. Synthèse CV
cv_df = pd.DataFrame(cv_scores)
cv_mean = cv_df.mean().round(4).sort_values()
cv_std = cv_df.std().round(4)
cv_summary = pd.DataFrame({
    'Modèle': cv_mean.index,
    'RMSE_moyen': cv_mean.values,
    'RMSE_std': [cv_std[m] for m in cv_mean.index]
}).reset_index(drop=True)
print("\n📊 Résultats CV")
print(cv_summary)
cv_summary.to_csv('results_oil_price/cv_results.csv', index=False)

best_model = cv_mean.idxmin()
best_rmse_cv = cv_mean.min()
print(f"\n🏆 Meilleur modèle : {best_model} (RMSE CV = {best_rmse_cv:.4f})")

# 9. Ré-entraînement sur train complet et évaluation test
print(f"\n🔄 Ré-entraînement de {best_model} sur l'ensemble d'entraînement complet...")

if best_model == 'ARIMA':
    model_final = train_arima(y_train, len(y_test))
    pred_test = model_final.predict(n_periods=len(y_test))
elif best_model == 'SARIMA':
    model_final = train_sarima(y_train, len(y_test))
    pred_test = model_final.predict(n_periods=len(y_test))
elif best_model == 'LSTM':
    model_final = build_lstm((SEQUENCE_LENGTH, 1))
    model_final.fit(X_seq_train_scaled, y_seq_train_scaled, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    pred_scaled = model_final.predict(X_seq_test_scaled).flatten()
    pred_test = scaler_seq_y.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
    pred_test = pd.Series(pred_test, index=dates_seq_test)
    common_dates = pred_test.index.intersection(y_test.index)
    y_test_aligned = y_test.loc[common_dates]
    pred_test_aligned = pred_test.loc[common_dates]
    rmse_test = np.sqrt(mean_squared_error(y_test_aligned, pred_test_aligned))
    mae_test = mean_absolute_error(y_test_aligned, pred_test_aligned)
    mape_test = mean_absolute_percentage_error(y_test_aligned, pred_test_aligned) * 100
    errors_test = y_test_aligned - pred_test_aligned
elif best_model == 'GRU':
    model_final = build_gru((SEQUENCE_LENGTH, 1))
    model_final.fit(X_seq_train_scaled, y_seq_train_scaled, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    pred_scaled = model_final.predict(X_seq_test_scaled).flatten()
    pred_test = scaler_seq_y.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
    pred_test = pd.Series(pred_test, index=dates_seq_test)
    common_dates = pred_test.index.intersection(y_test.index)
    y_test_aligned = y_test.loc[common_dates]
    pred_test_aligned = pred_test.loc[common_dates]
    rmse_test = np.sqrt(mean_squared_error(y_test_aligned, pred_test_aligned))
    mae_test = mean_absolute_error(y_test_aligned, pred_test_aligned)
    mape_test = mean_absolute_percentage_error(y_test_aligned, pred_test_aligned) * 100
    errors_test = y_test_aligned - pred_test_aligned
elif best_model == 'XGBoost':
    model_final = train_xgboost(X_train_scaled, y_train_scaled)
    pred_scaled = model_final.predict(X_test_scaled)
    pred_test = scaler_y.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
    mae_test = mean_absolute_error(y_test, pred_test)
    mape_test = mean_absolute_percentage_error(y_test, pred_test) * 100
    errors_test = y_test - pred_test
elif best_model == 'LightGBM':
    model_final = train_lightgbm(X_train_scaled, y_train_scaled)
    pred_scaled = model_final.predict(X_test_scaled)
    pred_test = scaler_y.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
    mae_test = mean_absolute_error(y_test, pred_test)
    mape_test = mean_absolute_percentage_error(y_test, pred_test) * 100
    errors_test = y_test - pred_test
elif best_model == 'Prophet':
    prophet_train = train[['Price']].reset_index()
    prophet_train.columns = ['ds', 'y']
    model_final = train_prophet(prophet_train)
    future = model_final.make_future_dataframe(periods=len(y_test))
    forecast = model_final.predict(future)
    pred_test = forecast['yhat'].values[-len(y_test):]
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
    mae_test = mean_absolute_error(y_test, pred_test)
    mape_test = mean_absolute_percentage_error(y_test, pred_test) * 100
    errors_test = y_test - pred_test
elif best_model == 'Stacking':
    split2 = int(0.8 * len(X_train_scaled))
    X_tr2, y_tr2 = X_train_scaled[:split2], y_train_scaled[:split2]
    X_val2, y_val2 = X_train_scaled[split2:], y_train_scaled[split2:]
    stacking_models = train_stacking(X_tr2, y_tr2, X_val2, y_val2)
    model_final = stacking_models  # <--- IMPORTANT : assigner model_final
    pred1 = stacking_models['models'][0].predict(X_test_scaled)
    pred2 = stacking_models['models'][1].predict(X_test_scaled)
    pred3 = stacking_models['models'][2].predict(X_test_scaled).flatten()
    meta_X = np.column_stack([pred1, pred2, pred3])
    pred_scaled = stacking_models['meta'].predict(meta_X)
    pred_test = scaler_y.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
    mae_test = mean_absolute_error(y_test, pred_test)
    mape_test = mean_absolute_percentage_error(y_test, pred_test) * 100
    errors_test = y_test - pred_test
else:
    raise ValueError(f"Modèle {best_model} non implémenté")

# Pour les modèles non-LSTM/GRU, on a déjà les métriques calculées.
if best_model not in ['LSTM', 'GRU']:
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
    mae_test = mean_absolute_error(y_test, pred_test)
    mape_test = mean_absolute_percentage_error(y_test, pred_test) * 100
    errors_test = y_test - pred_test

print(f"\n📈 Performance sur test : RMSE={rmse_test:.4f}, MAE={mae_test:.4f}, MAPE={mape_test:.2f}%")

# Calcul des erreurs de prévision pour la gestion des risques
std_error = errors_test.std()
var_95 = np.percentile(errors_test, (1 - VAR_CONFIDENCE) * 100)
var_95_positive = np.percentile(errors_test, VAR_CONFIDENCE * 100)

print(f"\n📊 Métriques de risque :")
print(f"   Écart-type des erreurs : {std_error:.4f} $")
print(f"   VaR 95% (pertes)       : {var_95:.4f} $")
print(f"   VaR 95% (gains)        : {var_95_positive:.4f} $")

# Graphique des prévisions sur test
plt.figure(figsize=(14,6))
if best_model in ['LSTM', 'GRU']:
    plt.plot(y_test_aligned.index, y_test_aligned, 'k-', label='Réel')
    plt.plot(pred_test_aligned.index, pred_test_aligned, 'r--', label=f'{best_model} (prévision)')
else:
    plt.plot(y_test.index, y_test, 'k-', label='Réel')
    plt.plot(y_test.index, pred_test, 'r--', label=f'{best_model} (prévision)')
plt.title(f'Prévisions du prix WTI – Test final')
plt.xlabel('Date'); plt.ylabel('Prix (USD)')
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig('results_oil_price/forecast_test.png', dpi=150)
plt.show()

# 10. Prévision future à 30 jours
print("\n🔮 Génération des prévisions futures à 30 jours...")
future_dates = pd.date_range(start=y_test.index[-1] + timedelta(days=1), periods=FUTURE_DAYS, freq='D')

if best_model in ['ARIMA', 'SARIMA']:
    future_pred = model_final.predict(n_periods=FUTURE_DAYS)
    future_pred = pd.Series(future_pred, index=future_dates)
elif best_model == 'Prophet':
    future = model_final.make_future_dataframe(periods=FUTURE_DAYS)
    forecast = model_final.predict(future)
    future_pred = forecast['yhat'].values[-FUTURE_DAYS:]
    future_pred = pd.Series(future_pred, index=future_dates)
elif best_model in ['LSTM', 'GRU']:
    last_sequence = X_seq_test_scaled[-1]
    future_pred_scaled = []
    for _ in range(FUTURE_DAYS):
        pred = model_final.predict(last_sequence.reshape(1, SEQUENCE_LENGTH, 1), verbose=0).flatten()[0]
        future_pred_scaled.append(pred)
        last_sequence = np.roll(last_sequence, -1, axis=0)
        last_sequence[-1] = pred
    future_pred = scaler_seq_y.inverse_transform(np.array(future_pred_scaled).reshape(-1,1)).flatten()
    future_pred = pd.Series(future_pred, index=future_dates)
elif best_model == 'Stacking':
    # Récupérer les modèles de base et le méta-modèle
    stacking_models = model_final
    model1, model2, model3 = stacking_models['models']
    meta_model = stacking_models['meta']
    
    # Historique initial : les dernières valeurs réelles (au moins 20)
    history = list(y_test.values[-min(20, len(y_test)):])
    future_pred = []
    
    # Vérifier le nombre de features attendu par le scaler
    n_features_expected = scaler_X.n_features_in_
    print(f"ℹ️ Le scaler attend {n_features_expected} features.")
    
    # Construire le vecteur de features dans le bon ordre (celui de X_train)
    # Récupérer les noms des colonnes de X_train pour être sûr de l'ordre
    feature_names = X_train.columns.tolist()  # Liste ordonnée
    print(f"Ordre des features : {feature_names}")
    
    for i in range(FUTURE_DAYS):
        # Calculer toutes les features à partir de l'historique
        # Initialiser un dictionnaire avec des valeurs par défaut (0)
        feat_dict = {name: 0.0 for name in feature_names}
        
        # Remplir les lags
        if len(history) >= 1:
            feat_dict['lag_1'] = history[-1]
        if len(history) >= 2:
            feat_dict['lag_2'] = history[-2]
        if len(history) >= 3:
            feat_dict['lag_3'] = history[-3]
        if len(history) >= 4:
            feat_dict['lag_4'] = history[-4]
        if len(history) >= 5:
            feat_dict['lag_5'] = history[-5]
        
        # Moyennes mobiles et écarts-types
        if len(history) >= 5:
            feat_dict['ma_5'] = np.mean(history[-5:])
            feat_dict['std_5'] = np.std(history[-5:])
        if len(history) >= 10:
            feat_dict['ma_10'] = np.mean(history[-10:])
            feat_dict['std_10'] = np.std(history[-10:])
        if len(history) >= 20:
            feat_dict['ma_20'] = np.mean(history[-20:])
            feat_dict['std_20'] = np.std(history[-20:])
        
        # Rendement
        if len(history) >= 2 and history[-2] != 0:
            feat_dict['return'] = (history[-1] - history[-2]) / history[-2]
        
        # Créer le vecteur dans l'ordre des colonnes
        features = np.array([feat_dict[name] for name in feature_names]).reshape(1, -1)
        
        # Normaliser
        features_scaled = scaler_X.transform(features)
        
        # Prédictions des modèles de base
        pred1 = model1.predict(features_scaled)[0]
        pred2 = model2.predict(features_scaled)[0]
        pred3 = model3.predict(features_scaled).flatten()[0]
        meta_features = np.array([[pred1, pred2, pred3]])
        pred_scaled = meta_model.predict(meta_features)[0]
        
        # Inverser la normalisation
        pred = scaler_y.inverse_transform([[pred_scaled]])[0,0]
        future_pred.append(pred)
        history.append(pred)
    
    future_pred = pd.Series(future_pred, index=future_dates)
else:
    print("⚠️ Prévision future non disponible pour ce modèle.")
    future_pred = []

if len(future_pred) > 0:
    plt.figure(figsize=(14,6))
    plt.plot(train.index[-60:], train['Price'].iloc[-60:], 'b-', label='Historique')
    plt.plot(test.index, test['Price'], 'k-', label='Test')
    # Personnaliser la légende pour Stacking
    if best_model == 'Stacking':
        plt.plot(future_dates, future_pred, 'r--', label='30-day ahead price forecast using the Stacking ensemble')
    else:
        plt.plot(future_dates, future_pred, 'r--', label=f'Prévision {best_model} (30j)')
    plt.axvline(y_test.index[-1], color='gray', linestyle=':', alpha=0.7)
    plt.title('Prévision du prix à 30 jours')
    plt.xlabel('Date'); plt.ylabel('Prix (USD)')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig('results_oil_price/future_forecast.png', dpi=150)
    plt.show()

    # Sauvegarder les prévisions futures
    future_df = pd.DataFrame({'Date': future_dates, 'Prévision': future_pred})
    future_df.to_csv('results_oil_price/future_forecast.csv', index=False)

# 11. Rapport HTML interactif (inchangé)
print("\n📄 Génération du rapport HTML...")
# ... (le code du rapport HTML est identique à celui fourni précédemment, je ne le répète pas pour économiser de l'espace, mais vous devez le conserver) ...

# 12. Archivage des résultats
print("\n📦 Création de l'archive ZIP...")
with zipfile.ZipFile('results_oil_price.zip', 'w') as zipf:
    for root, dirs, files in os.walk('results_oil_price'):
        for file in files:
            zipf.write(os.path.join(root, file), arcname=file)
print("✅ Archive créée : results_oil_price.zip")

# 13. Publication GitHub (optionnel) - inchangé
print("\n🐙 Préparation pour l'envoi sur GitHub...")
# ... (code GitHub existant) ...

print("\n🎉 Toutes les étapes sont terminées avec succès !")
