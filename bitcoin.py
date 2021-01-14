import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, max_error
from sklearn.model_selection import RandomizedSearchCV
import joblib


def load_data():
    df = pd.read_csv('data/bitcoin_data.zip')
    df.dropna(thresh = 7, inplace = True)
    df['day_change'] = df['Open'] - df['Close']
    df['label'] = np.where(df['day_change'] > 0, 'up', 'down')
    df.set_index('Timestamp', inplace= True)
    df.drop(['Low', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price', 'day_change', 'Close'], axis=1, inplace=True)
    df = df[2000000:]
    return df
df = load_data()
y_1 = df.pop('label')
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
    return grid_grb

# grb_fit = fit_grb(X_train, y_train)
# grb_pred = grb_fit.predict(X_test)

# grb_rmse = mean_squared_error(y_test, grb_pred, squared=False)  
# grb_max_er = max_error(y_test, grb_pred)
# print(f'mse: {grb_rmse}, max_error: {grb_max_er}')

def fit_random_forest(X_train, y_train):
    rf = RandomForestClassifier(warm_start = True, n_jobs = -1)
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop =1000, num = 5)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                                    n_iter = 25, cv = 3, verbose=2, random_state=42)
    print('fitting random forest fit 1')
    rf_random.fit(X_train[0:150000], y_train[0:150000])
    print('fitting random forest fit 2')
    rf_random.fit(X_train[150000:300000], y_train[150000:300000])
    print('fitting random forest fit 3')
    rf_random.fit(X_train[300000:450000],  y_train[300000:450000])
    print('fitting random forest fit 4')
    rf_random.fit(X_train[450000:700000], y_train[450000:700000])
    rf_random.fit(X_train[700000:950000], y_train[700000:950000])
    rf_random.fit(X_train[950000:1000000], y_train[950000:1000000])
    rf_random.fit(X_train[1000000:], y_train[1000000:])
    return rf_random

# rf_fit = fit_random_forest(X_train, y_train)
# rf_pred = rf_fit.predict(X_test)
# mse = mean_squared_error(y_test, rf_pred, squared=False)  
# max_er = max_error(y_test, rf_pred)
# joblib.dump(rf_fit, 'grid_rf.pkl')