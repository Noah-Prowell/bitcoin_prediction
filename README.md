# Bitcoin Prediction


 - I started this project with the idea to predict the price of bitcoin for my own personal use.  My data consists of pricing data for bitcoin which includes the open and closing prices, volumes, high's, low's and so on.   
 - The original data is the proce of bitcoin every 60 seconds since the start of bitcoin(2012).  This was a lot of data points with a lot of repeats and tiny increments.  Now since bitcoin does not technically have a closing price, since the price never stops changing unlike the stock market, I used the price every 24 hours as the opening and closing prices of Bitcoin.
 - I needed to simplify my problem from being able to predict the exact price of Bitcoin 24 hours later to a classification problem with two classes. 
 - I then used the closing prices and the high's to train a Random Forest model to classify whether the price of bitcoin would be going up or down within the next 24 hours.  This worked rather well as I was able to get an accuracy of approximately 80%. But I wasnt done yet.
 - After this model was done I decided I wanted a model that could generally predict Bitcoin in the near future.  And so I created an ARIMA model that is trained on the closing price of Bitcoin.
 -  Exerything can be seen here https://share.streamlit.io/noah-prowell/bitcoin_prediction/main/app.py on my live streamlit website.
 -  *On a side note about streamlit.  I am a huge fan of this because they make it super easy to deploy live machine learning apps without having to use flask and HTML which for someone like me that is a huge advantage.


