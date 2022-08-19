#!/usr/bin/env python
# coding: utf-8

# In[235]:


#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


# In[236]:


#reading dataset into dataframe as data1
data1 = pd.read_csv("C:/Users/Lenovo/Desktop/time series project summer/vaccinations.csv")
data1.head(5)


# In[237]:


#checking dimention of data1
data1.shape


# In[238]:


#CHECKING FOR NULL VALUES in data1
data1.isnull().sum()


# In[239]:


#Replacing NA by 0
data1 = data1.fillna(0)
data1


# In[240]:


#CHECKING FOR NULL VALUES IN EACH COLUMNS
data1.isnull().sum()


# In[241]:


#CHECKING COUNT OF UNIQUE VALUES IN COLUMNS
data1['date'].value_counts()


# In[242]:


#DROPPING THE locations and ISO code COLUMNs FROM data1
data1.drop(['location','iso_code'], axis=1, inplace=True)
data1.head()


# In[243]:


data1.shape


# In[244]:


#finding list of numerical value columns into list num_col
num_col = [i for i in data1.columns if data1.dtypes[i]!='object']
num_col


# In[245]:


#sorting dataframe by Date in ascending order
data1 = data1.sort_values(by = 'date')
data1


# In[ ]:





# In[246]:


#grouping dataframe by Date and summing them date wise data1 as data2
data2 = data1.groupby(['date'], as_index = False).sum()
data2


# In[247]:


#copying the date column
col_date = data2['date'].copy()


# In[248]:


len(num_col)


# In[249]:


np.random.seed(10)


# In[250]:


#now we make our dataframe data2 continuous by adding random number using unniform(0,1) and name it as added_df
added_df = data2.iloc[:,1:].add(np.random.rand(221, 9))
added_df


# In[251]:


#adding col_date column in added_df dataframe at right most column
added_df['Date'] = col_date
added_df


# In[252]:


#making list  of column names of added_df dataframe as cols
cols=list(added_df.columns)



#taking total_vaccinations and date columns in new dataframe df_date_tot to fit the model in this dataframe to forecast total_vaccinations
df_date_tot = added_df[[cols[4]] + [cols[-1]]]
df_date_tot


# In[253]:


#plotting the df_date_tot dataframe date vs total_vaccinations all over the wworld
plt.figure(figsize=(3,3),dpi=300)
plt.plot(df_date_tot['Date'],df_date_tot['daily_vaccinations'])
plt.show()
plt.subplot(212)
plt.hist(df_date_tot['daily_vaccinations'])
plt.show()


# In[254]:


# splitting dataframe by row index into two parts
part_1 = df_date_tot.iloc[:111,:]
part_2 = df_date_tot.iloc[111:,:]
df_date_tot.describe()


# In[255]:


part_1.shape


# In[256]:


part_2.shape


# In[257]:


part_1.var()


# In[258]:


part_2.var()


# In[259]:


1.477432e+15-1.873548e+14


# In[260]:


df_daily_vaccinations = df_date_tot.copy()
df_daily_vaccinations.shape


# In[261]:


#box cox method for variance stablization
from scipy.stats import boxcox
df_daily_vaccinations['daily_vaccinations'], lam = boxcox(df_daily_vaccinations['daily_vaccinations'])
print('Lambda: %f' % lam)
# line plot
plt.figure(figsize=(3,3),dpi=300)
plt.plot(df_daily_vaccinations['Date'],df_daily_vaccinations['daily_vaccinations'])
plt.show()
# histogram
plt.subplot(212)
plt.hist(df_daily_vaccinations['daily_vaccinations'])
plt.show()


# In[262]:


df_daily_vaccinations['daily_vaccinations']


# In[263]:


train = df_daily_vaccinations['daily_vaccinations'][:154]
test = df_daily_vaccinations['daily_vaccinations'][154:]


# In[264]:


plt.figure(figsize=(3,3),dpi=300)
plt.plot(df_daily_vaccinations['Date'],df_daily_vaccinations['daily_vaccinations'])
plt.show()


# In[265]:


#testing the stationarity by DF test which results it as stationary time series
#by looking at the graph and adf test it is clearly  non_stationary.


from statsmodels.tsa.stattools import adfuller
result = adfuller(df_daily_vaccinations['daily_vaccinations'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# In[266]:


# KPSS test
from statsmodels.tsa.stattools import kpss
result = kpss(df_daily_vaccinations['daily_vaccinations'])
print(result)
#shows series is non stationary
    


# In[267]:


#ACF plot upto lag 50

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df_daily_vaccinations['daily_vaccinations'], lags = 50)
plt.show()


# In[268]:


#first order differencing of raw data
df_daily_vaccinations['daily_vaccinations_with_diff_1'] = df_daily_vaccinations['daily_vaccinations'] - df_daily_vaccinations['daily_vaccinations'].shift(1)
df_daily_vaccinations


# In[269]:


#ploting the time series after first  order differning
plt.figure(figsize=(5,5),dpi=300)
plt.plot(df_daily_vaccinations['Date'],df_daily_vaccinations['daily_vaccinations_with_diff_1'])
plt.show()


# In[270]:


#acf plot of dataframe of diff 1
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df_daily_vaccinations['daily_vaccinations_with_diff_1'][1:], lags = 50)
plt.show()


# In[271]:


#pacf plot of dataframe of diff 1
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(df_daily_vaccinations['daily_vaccinations_with_diff_1'][1:], lags = 50)
plt.show()


# In[272]:


#DF test on data with diff 1
from statsmodels.tsa.stattools import adfuller
result = adfuller(df_daily_vaccinations['daily_vaccinations_with_diff_1'][1:])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
#shows non stationary


# In[273]:


# KPSS test
from statsmodels.tsa.stattools import kpss
result = kpss(df_daily_vaccinations['daily_vaccinations_with_diff_1'][1:])
print(result)
#shows series is non stationary


# In[ ]:





# In[274]:


#second order differencing of raw data
df_daily_vaccinations['daily_vaccinations_with_diff_2'] = df_daily_vaccinations['daily_vaccinations_with_diff_1'] - df_daily_vaccinations['daily_vaccinations_with_diff_1'].shift(1)
df_daily_vaccinations


# In[275]:


#ploting the time series after second  order differning
plt.figure(figsize=(5,5),dpi=500)
plt.plot(df_daily_vaccinations['Date'],df_daily_vaccinations['daily_vaccinations_with_diff_2'])
plt.show()
# histogram
plt.subplot(212)
plt.hist(df_daily_vaccinations['daily_vaccinations_with_diff_2'])
plt.show()


# In[276]:


#acf plot of dataframe of diff 2
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df_daily_vaccinations['daily_vaccinations_with_diff_2'][2:], lags = 50)
plt.show()


# In[277]:


#pacf plot of dataframe of diff 2
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(df_daily_vaccinations['daily_vaccinations_with_diff_2'][2:], lags = 50)
plt.show()


# In[278]:


#df test for data with diff 2
from statsmodels.tsa.stattools import adfuller
result = adfuller(df_daily_vaccinations['daily_vaccinations_with_diff_2'][2:])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
#shows series is stationary


# In[279]:


# KPSS test
from statsmodels.tsa.stattools import kpss
result = kpss(df_daily_vaccinations['daily_vaccinations_with_diff_2'][2:])
print(result)
#shows series is stationary


# In[280]:


df_date_tot


# In[335]:


#fitting arima models
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMAResults
import re
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
import pylab 
import scipy.stats as stats
from scipy.stats import anderson
from scipy.stats import shapiro
from scipy.stats import normaltest
#ARIMA Model
def arima(p,d,q):
    model = ARIMA(train, order=(p,d,q))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    print(ARMAResults.cov_params(model_fit))
    diag_elt = np.sqrt(ARMAResults.cov_params(model_fit).diagonal())
    D = np.diag(diag_elt)
    corr_mat = np.linalg.inv(D).dot(ARMAResults.cov_params(model_fit)).dot(np.linalg.inv(D))
    print(corr_mat)
    
    
    # Plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    fig, ax = plt.subplots(1,2)
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.show()
    plt.savefig('C:/Users/Lenovo/Desktop/resdi_plots.png',dpi=100)
    
    # q-q plot of residuals
    plt.subplot(1,1,1)
    qqplot(model_fit.resid, line='r', ax=plt.gca())
    plt.show()
    plt.savefig('C:/Users/Lenovo/Desktop/qq_plots.png',dpi=500)
    
    #anderson darling test of normality of residual distribution
    result = anderson(model_fit.resid)
    print('Statistic: %.3f' % result.statistic)
    p = 0
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < result.critical_values[i]:
            print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
        else:
            print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
            
    # Shapiro-Wilk Test of normality of residual distribution
    stat, p = shapiro(model_fit.resid)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')
    
    # Mann-Whitney U test
    from numpy.random import randn
    from scipy.stats import mannwhitneyu
    data1 = randn(67)
    stat, p = mannwhitneyu(data1, model_fit.resid)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')
    
    
    #stats.probplot(model_fit.resid, dist="norm", plot=plt)
    
    
    # acf plot of residuals
    plot_acf(residuals, lags = 50)
    plt.show()
    plt.savefig('C:/Users/Lenovo/Desktop/acf_resid.png',dpi=100)
    # Forecast
    fc, se, conf = model_fit.forecast(67, alpha=0.05)  # 95% conf
    
    for i in range(len(test)):
        print('>Predicted=%.3f, Actual=%.3f' % (fc.tolist()[i], test.reset_index(drop=True).tolist()[i]))
    
    #ljung box test
    print(sm.stats.acorr_ljungbox(residuals, lags=[1,2,3,4,5,10,12,14,15,17], return_df=True)) #if statistics p value is less than critical value than reject H0
    #means residuals are correlated

    # Make as pandas series
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(conf[:, 0], index=test.index)
    upper_series = pd.Series(conf[:, 1], index=test.index)

    # Plot
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(train, label='training')
    plt.plot(test, label='actual')
    plt.plot(fc_series, label='forecast')
    plt.fill_between(lower_series.index, lower_series, upper_series, 
                     color='k', alpha=.15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.savefig('C:/Users/Lenovo/Desktop/forecast_f.png',dpi=100)
    plt.show()
    
    


# In[282]:


arima(1,1,1) #bad


# In[283]:


arima(2,1,1) #bad


# In[284]:


arima(1,1,2) #bad


# In[285]:


arima(2,1,2) #second best


# In[286]:


arima(2,2,4)


# In[287]:


arima(3,1,3) #third best


# In[288]:


arima(1,2,1) #less AIC than 2,1,2 but not all variables are signifi


# In[289]:


arima(2,2,1) #bad


# In[290]:


arima(1,2,2) #bad


# In[291]:


arima(2,2,2) #bad


# In[336]:


arima(2,2,3) #best


# In[293]:


arima(0,1,0)


# In[294]:


import pmdarima as pm
model = pm.auto_arima(train, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=4, max_q=4, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())


# In[295]:


model.plot_diagnostics(figsize=(7,5))
plt.show()


# In[333]:


arima(4,2,4)


# In[297]:


arima(1,2,1)


# In[298]:


arima(1,2,0)


# In[299]:


arima(0,2,2)


# In[300]:


arima(1,2,2)


# In[301]:


arima(2,2,2)


# In[302]:


arima(2,2,1)


# In[303]:


arima(2,2,0)


# In[304]:


arima(0,2,3)


# In[305]:


arima(1,2,3)


# In[306]:


arima(2,2,3)


# In[307]:


arima(3,2,3)


# In[308]:


arima(3,2,2)


# In[309]:


arima(3,2,1)


# In[310]:


arima(2,1,2)


# In[317]:


arima(1,2,4)


# In[ ]:





# In[ ]:





# In[ ]:




