# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 19:09:32 2025

@author: Bhavana
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

investment = pd.read_csv(r'D:\Data Science and Machine Learning\March\26th_MultipleLinearRegression\Investment.csv')

X = investment.iloc[:,:4]
Y = investment.iloc[:,-1]

# Called Imputation/Transformer..What is ML Transformer??
X = pd.get_dummies(X,dtype=int)


X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,Y,test_size=.2,random_state=0)

# regressor is the MODEL
regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)

Y_Pred = regressor.predict(X_Test)

m_slope = regressor.coef_
c_intercept = regressor.intercept_
df = pd.DataFrame({'Slope':m_slope, 'Intercept':c_intercept})
print(df)

# Instead of 1 try to add intercept 42467
#DummyENcoder, One Hot nco etc
X = np.append(arr = np.ones((50,1)).astype(int),values= X, axis = 1)

# Feature elimi
#Recur Elimi
#Backward elim
# Forw elimi
X_opt = X[:,[0,1,2,3,4,5]]
# Ordinary Least Squares
#endog --> 
#exog -->
regressor_OLS = sm.OLS(endog = Y, exog=X_opt).fit()
regressor_OLS.summary()


# Elimi based on p value and signifi value...990 is highest..

# Feature elimi
#Recur Elimi
#Backward elim
# Forw elimi
# Why did we remove last column???
X_opt = X[:,[0,1,2,3,5]]
# Ordinary Least Squares
regressor_OLS = sm.OLS(endog = Y, exog=X_opt).fit()
regressor_OLS.summary()


X_opt = X[:,[0,1,2,3]]
# Ordinary Least Squares
regressor_OLS = sm.OLS(endog = Y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,3]]
# Ordinary Least Squares
regressor_OLS = sm.OLS(endog = Y, exog=X_opt).fit()
regressor_OLS.summary()


X_opt = X[:,[0,1]]
# Ordinary Least Squares
regressor_OLS = sm.OLS(endog = Y, exog=X_opt).fit()
regressor_OLS.summary()
