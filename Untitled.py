#!/usr/bin/env python
# coding: utf-8

# ## Hourly Time series forcasting using Facebook's Prophet

# In this notebook we will use facebook's prophet package to forecast hourly energy use.The data we will be using is hourly power consumption data from PJM. Energy consumption has some unique characteristics. it will be intresting to see how prophet picks them up.PJM_East_hourly which has data from 20002-2018 for the entire east region.

# In[60]:


# !pip install Prophet


# In[13]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet

from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')
plt.style.use('fivethirtyeight')

#defines a function called mean_absolute_percentage_error (MAPE)
#that calculates the Mean Absolute Percentage Error between actual values (y_true) and predicted values (y_pred)
def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[25]:


#importing data
import os
os.getcwd()
pjm_data = pd.read_csv('PJME_hourly.csv',index_col=[0],parse_dates=[0])
pjm_data.head()


# In[26]:


len(pjm_data)


# In[28]:


# plotting the data 
color_pal=sns.color_palette()

pjm_data.plot(style='.',
             figsize=(10,5),
             ms=1,
             color=color_pal[0],
             title='PJM MW')

plt.show()


# here we can see this data is going from 2002  all the way to 2018.and you can see that it has some trends in the data.and as we dig down deeper, we're going to see even has it at the hourly level,some trends we can pick upon.In order to visualize, data before modelling, we can use some time series fetaures.

# ## Time Series Features

# In[29]:


from pandas.api.types import CategoricalDtype

cat_type=CategoricalDtype(categories=['Monday','Tuesday',
                                      'Wednesday','Thursday',
                                      'Friday','Saturday','Sunday'],ordered=True)

def create_features(df, label=None):
    """
    Creates time series features from datetime index.
    """
    df = df.copy()
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekday'] = df['date'].dt.day_name()
    df['weekday'] = df['weekday'].astype(cat_type)
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    df['date_offset'] = (df.date.dt.month*100 + df.date.dt.day - 320)%1300

    df['season'] = pd.cut(df['date_offset'], [0, 300, 602, 900, 1300], 
                          labels=['Spring', 'Summer', 'Fall', 'Winter']
                   )
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear','weekday',
           'season']]
    if label:
        y = df[label]
        return X, y
    return X

X,y =create_features(pjm_data,label='PJME_MW')
features_and_target=pd.concat([X,y],axis=1)


# In[30]:


fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=features_and_target.dropna(),
            x='weekday',
            y='PJME_MW',
            hue='season',
            ax=ax,
            linewidth=1)
ax.set_title('Power Use MW by Day of Week')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Energy (MW)')
ax.legend(bbox_to_anchor=(1, 1))
plt.show()


# ## Train/Test Split

# In[31]:


split_date='1-Jan-2015'
pjm_train=pjm_data.loc[pjm_data.index<= split_date].copy()
pjm_test = pjm_data.loc[pjm_data.index > split_date].copy()

# Plot train and test so you can see where we have split
pjm_test \
    .rename(columns={'PJME_MW': 'TEST SET'}) \
    .join(pjm_train.rename(columns={'PJME_MW': 'TRAINING SET'}),
          how='outer') \
    .plot(figsize=(10, 5), title='PJM East', style='.', ms=1)
plt.show()


# ## Simple Prophet Model

# Prophet model expects the dataset to be named a specific way. We will rename our dataframe columns before feeding it into the model.
# Datetime column named: ds
# target : y

# In[32]:


# Format data for prophet model using ds and y
pjm_train_prophet = pjm_train.reset_index() \
    .rename(columns={'Datetime':'ds',
                     'PJME_MW':'y'})


# In[33]:


get_ipython().run_cell_magic('time', '', 'model = Prophet()\nmodel.fit(pjm_train_prophet)\n')


# In[34]:


# Predict on test set with model
pjm_test_prophet = pjm_test.reset_index() \
    .rename(columns={'Datetime':'ds',
                     'PJME_MW':'y'})

pjm_test_fcst = model.predict(pjm_test_prophet)


# In[35]:


pjm_test_fcst.head()


# In[36]:


fig, ax = plt.subplots(figsize=(10, 5))
fig = model.plot(pjm_test_fcst, ax=ax)
ax.set_title('Prophet Forecast')
plt.show()


# In[37]:


fig = model.plot_components(pjm_test_fcst)
plt.show()


# ## compare Forecast to Actuals

# In[38]:


# Plot the forecast with the actuals
f, ax = plt.subplots(figsize=(15, 5))
ax.scatter(pjm_test.index, pjm_test['PJME_MW'], color='r')
fig = model.plot(pjm_test_fcst, ax=ax)


# In[42]:


fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(pjm_test.index, pjm_test['PJME_MW'], color='r')
fig = model.plot(pjm_test_fcst, ax=ax)
start_date = pd.to_datetime('2015-01-01')
end_date = pd.to_datetime('2015-02-01')
ax.set_xbound(lower=start_date,
              upper= end_date)
ax.set_ylim(0, 60000)
plot = plt.suptitle('January 2015 Forecast vs Actuals')


# In[45]:


# Plot the forecast with the actuals
f, ax = plt.subplots(figsize=(15, 5))
ax.scatter(pjm_test.index, pjm_test['PJME_MW'], color='r')
fig = model.plot(pjm_test_fcst, ax=ax)
s = pd.to_datetime('2015-01-01')
e = pd.to_datetime('2015-08-01')
ax.set_xbound(lower=s, upper=e)
ax.set_ylim(0, 60000)
ax.set_title('First Week of January Forecast vs Actuals')
plt.show()


# ## Evaluate the model with Error Metrics

# In[46]:


np.sqrt(mean_squared_error(y_true=pjm_test['PJME_MW'],
                   y_pred=pjm_test_fcst['yhat']))


# In[47]:


mean_absolute_error(y_true=pjm_test['PJME_MW'],
                   y_pred=pjm_test_fcst['yhat'])


# In[48]:


mean_absolute_percentage_error(y_true=pjm_test['PJME_MW'],
                   y_pred=pjm_test_fcst['yhat'])


# ## Adding Holidays
# 
# Next we will see if adding holiday indicators will help the accuracy of the model. Prophet comes with a Holiday Effects parameter that can be provided to the model prior to training.
# 
# We will use the built in pandas USFederalHolidayCalendar to pull the list of holidays

# In[49]:


from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

cal = calendar()


holidays = cal.holidays(start=pjm_data.index.min(),
                        end=pjm_data.index.max(),
                        return_name=True)
holiday_df = pd.DataFrame(data=holidays,
                          columns=['holiday'])
holiday_df = holiday_df.reset_index().rename(columns={'index':'ds'})


# In[50]:


get_ipython().run_cell_magic('time', '', 'model_with_holidays = Prophet(holidays=holiday_df)\nmodel_with_holidays.fit(pjm_train_prophet)\n')


# In[51]:


# Predict on training set with model
pjm_test_fcst_with_hols = \
    model_with_holidays.predict(df=pjm_test_prophet)


# In[52]:


fig = model_with_holidays.plot_components(
    pjm_test_fcst_with_hols)
plt.show()


# In[54]:


fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(pjm_test.index, pjm_test['PJME_MW'], color='r')
fig = model.plot(pjm_test_fcst_with_hols, ax=ax)
l = pd.to_datetime('07-01-2015')
u = pd.to_datetime('07-07-2015')
ax.set_xbound(lower=l,
              upper=u)
ax.set_ylim(0, 60000)
plot = plt.suptitle('July 4 Predictions vs Actual')


# In[55]:


np.sqrt(mean_squared_error(y_true=pjm_test['PJME_MW'],
                   y_pred=pjm_test_fcst_with_hols['yhat']))


# In[56]:


mean_absolute_error(y_true=pjm_test['PJME_MW'],
                   y_pred=pjm_test_fcst_with_hols['yhat'])


# In[57]:


mean_absolute_percentage_error(y_true=pjm_test['PJME_MW'],
                   y_pred=pjm_test_fcst_with_hols['yhat'])


# ## Predict into the Future
# We can use the built in make_future_dataframe method to build our future dataframe and make predictions.

# In[58]:


future = model.make_future_dataframe(periods=365*24, freq='h', include_history=False)
forecast = model_with_holidays.predict(future)


# In[59]:


forecast[['ds','yhat']].head()


# In[ ]:




