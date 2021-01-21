import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import joblib
import streamlit as st
import pickle
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import power_transform
from datetime import datetime
import statsmodels.api as sm
import plotly.express as px

@st.cache
def read_in_data():
    df = pd.read_csv('data/bitcoin_try.csv')
    df.dropna(thresh = 7, inplace = True)
    df['day_change'] = df['Open'] - df['Close']
    df['label'] = np.where(df['day_change'] > 0, 'up', 'down')
    df.set_index('Timestamp', inplace= True)
    df.drop(['Low', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price', 'day_change', 'Close'], axis=1, inplace=True)
    df = df[::1400]
    return df
df = read_in_data()
model = joblib.load('grid_rf.sav')

st.write('''# Bitcoin Predictor''')
st.write('''Input the open to predict whether bitcoin will go up or down''')
open_p = st.number_input(label='Input Open Here')
high = df.High.iloc[-1]
# high = st.number_input(label='Input Current High Here')
data = {'Open':[open_p], 'High':[high]}
input_pred = pd.DataFrame(data)
if st.button('Predict'):
    try: 
        pred = model.predict(input_pred)
        if pred == 0:
            st.write('Predict Bitcoin going DOWN')
        else:
            st.write('Predict Bitcoin going UP!')
    except:
        'Type Error: Wrong Data'






def load_arima_data():
    df = pd.read_csv('data/bitcoin_try.csv')
    df = df.iloc[::1440]
    close = df.pop('Close')
    close_df = pd.DataFrame(close)
    return close_df

price = load_arima_data()


def get_price_24(df):
    price_24 = df.fillna(method = 'ffill', inplace=True)
    price_24 = df.set_index(pd.date_range(start='12-31-2011', end='12-26-2020', freq='D'))
    return price_24

price_24 = get_price_24(price)

price_model = ARIMA(price_24, order=(1, 1, 0)).fit()

date_in = st.text_input(label='Input Date to Predict to(format YYYY-MM-DD')
date_in = str(date_in)

if st.button('Arima Prediciton'):
    fig, ax = plt.subplots(1, figsize=(14, 4))
    ax.plot(price_24['2017':].index, price_24['2017':])
    fig = price_model.plot_predict('2020', f'{date_in}', 
                                    dynamic=True, ax=ax, plot_insample=False)

    ax.legend().get_texts()[1].set_text("95% Prediction Interval")
    ax.legend(loc="lower left")

    ax.set_title("Price Forecasts from ARIMA Model");
    st.pyplot(fig)




date = price_24.index
fig3 = px.line(price_24, x=date, y=price_24['Close'])
fig3.show()
st.plotly_chart(fig3)