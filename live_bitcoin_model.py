import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import joblib
import time




def build_model_and_ttt(data):
    X, X_h, y, y_h = train_test_split(X_1, y_1, test_size = .1)
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    return X_train, X_test, y_train, y_test, X_h, y_h



def fit_grb(X_train, y_train):
    grb_model = GradientBoostingClassifier(subsample = .25)
    params_grb = {'learning_rate': [.001, .0001, .005], 'n_estimators':[1500, 1700, 1850], 'max_depth': [6, 7, 8]}
    grid_grb = GridSearchCV(grb_model, params_grb)
    print('fitting grb model')
    start = time.time()
    grid_grb.fit(X_train, y_train)
    end = time.time()
    print(f'Time to Train {end} - {start}')
    return grid_grb


def fit_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_jobs = -1)
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop =2000, num = 15)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(2, 80, num = 15)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [10, 15, 17]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 3]
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
                                    n_iter = 200, cv = 5, verbose=2, random_state=42)
    print('fitting random forest')
    start = time.time()
    rf_random.fit(X_train, y_train)
    end = time.time()
    print(f'Time to Train {end} - {start}')
    return rf_random


if __name__ == '__main__':
    df = pd.read_json('full_data.json')
    y_1 = df.pop('label')
    X_1 = df
    X_train, X_test, y_train, y_test, X_h, y_h = build_model_and_ttt(df)
    # rf_fit = fit_random_forest(X_train, y_train)
    # rf_pred = rf_fit.predict(X_test)
    # mse = accuracy_score(y_test, rf_pred)  
    # print(rf_fit.best_params_)
    # print(f'The mse is {mse}')
    # max_er = max_error(y_test, rf_pred)


    grb_fit = fit_grb(X_train, y_train)
    grb_pred = grb_fit.predict(X_test)

    grb_rmse = accuracy_score(y_test, grb_pred)   
    # grb_max_er = max_error(y_test, grb_pred)
    print(f'mse: {grb_rmse}')
    joblib.dump(grb_fit, 'grid_grb.sav')