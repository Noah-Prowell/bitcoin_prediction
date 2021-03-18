#!/usr/bin/env python
# coding: utf-8

# In[16]:


import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from time import time, sleep
import boto3

# In[2]:


def transform_df(file_name):
    df = pd.read_json(file_name)
    df.drop('disclaimer', axis = 1, inplace = True)
    df.drop('time', axis = 1, inplace = True)
    df.drop(['updated', 'updatedISO'], inplace = True)
    diff = df.diff().rename(columns = {'bpi': 'Diff'}).Diff.values
    df['diff'] = diff
    df.fillna(0, inplace = True)
    df['label'] = np.where(df['diff'] > 0, True, False)
    df.drop('diff', axis = 1, inplace= True)
    return df


# In[3]:


# s = sched.scheduler(time.time, time.sleep)
def run_api():
    yesterday = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')
    get_ipython().system("curl 'https://api.coindesk.com/v1/bpi/historical/close.json?start=2011-12-31&end={yesterday}' > current_price.json")
    full_df = transform_df('current_price.json')
    full_df.to_json('full_data.json')
#     s.enter(86400, 1)


# In[4]:




# In[11]:
    

# In[ ]:
def upload_file(file_name, bucket, object_name=None):
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

while True:
    sleep(86400 - time() % 86400)
    run_api()
    upload_file('full_data.json', 'noah-bitcoin', 'full_data.json')








