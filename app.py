import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, max_error
from sklearn.model_selection import RandomizedSearchCV
import joblib
import streamlit as st
import pickle

@st.cache
def read_in_data():
    df = pd.read_csv('data/bitcoin_data.zip')
    df.dropna(thresh = 7, inplace = True)
    df['day_change'] = df['Open'] - df['Close']
    df['label'] = np.where(df['day_change'] > 0, 'up', 'down')
    df.set_index('Timestamp', inplace= True)
    df.drop(['Low', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price', 'day_change', 'Close'], axis=1, inplace=True)
    df = df[2000000:]
    return df
# df = read_in_data()
model = joblib.load('first_rf.pkl')

st.text('Input the open and the current high to predict whether bitcoin will go up or down')
open_p = st.number_input(label='Input Open Here')
high = st.number_input(label='Input Current High Here')
data = {'Open':[open_p], 'High':[high]}
input_pred = pd.DataFrame(data)
if st.button('Predict'):
    try: 
        model.predict(input_pred)
    except:
        'Type Error: Wrong Data'
else:
    st.write('Click me Boi')