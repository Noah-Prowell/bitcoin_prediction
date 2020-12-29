import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, max_error

def load_data():
    df = pd.read_csv('data/bitcoin_data.csv')
    df.dropna(thresh = 7, inplace = True)
    df['day_change'] = df['Open'] - df['Close']
    df.set_index('Timestamp', inplace= True)
    return df
df = load_data()
y_1 = df.pop('Close')
X_1 = df

def build_model_and_ttt(data):
    X, X_h, y, y_h = train_test_split(X_1, y_1, test_size = .1)
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    return X_train, X_test, y_train, y_test, X_h, y_h

X_train, X_test, y_train, y_test, X_h, y_h = build_model_and_ttt(df)

def fit_grb(X_train, y_train):
    grb_model = GradientBoostingRegressor(subsample = .25)
    params_grb = {'learning_rate': [.01,], 'n_estimators':[1000, 1200, 1500], 'max_depth': [4, 5, 6]}
    grid_grb = GridSearchCV(grb_model, params_grb)
    print('fitting grb model')
    grid_grb.fit(X_train, y_train)

grb_fit = fit_grb(X_train, y_train)
grb_pred = grb_fit.predict(X_test)

grb_rmse = mean_squared_error(y_test, grb_pred, squared=False)  
grb_max_er = max_error(y_test, grb_pred)
print(f'mse: {grb_rmse}, max_error: {grb_max_er}')

def fit_random_forest(X_train, y_train):
    rf = RandomForestRegressor(warm_start = True)
    print('fitting random forest')
    rf.fit(X_train[0:562000], y_train[0:562000])
    rf.fit(X_train[562000:1124000], y_train[562000:1124000])
    rf.fit(X_train[1124000:2248000],  y_train[1124000:2248000])
    rf.fit(X_train[2248000:], y_train[2248000:])
    return rf

rf_fit = fit_random_forest(X_train, y_train)
rf_pred = rf_fit.predict(X_test)
mse = mean_squared_error(y_test, rf_pred, squared=False)  
max_er = max_error(y_test, rf_pred)