# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 19:37:06 2025

@author: Hithaardh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

salary = pd.read_csv(r'D:\Data Science and Machine Learning\March\18thMachineLearning\Salary_data.csv')


X = salary.iloc[:,0]

y = salary.iloc[:,-1]

from sklearn.model_selection import train_test_split

X_Train,X_Test,y_Train,y_Test = train_test_split(X,y,test_size=.2,random_state=0)

X_Train = X_Train.values.reshape(-1,1)
X_Test = X_Test.values.reshape(-1,1)



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_Train,y_Train)

y_Pred = regressor.predict(X_Test)



plt.scatter(X_Test,y_Test,color='red')
plt.plot(X_Train, regressor.predict(X_Train),color='blue')
plt.title('Salary vs Experience TEST SET')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


plt.scatter(X_Train,y_Train,color='red')
plt.plot(X_Test, regressor.predict(X_Test),color='blue')
plt.title('Salary vs Experience TEST SET')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


print(f'Coefficient of Variance = {regressor.coef_}')
coef = regressor.coef_
print(f'Coefficient of Variance = {regressor.intercept_}')
intercept = regressor.intercept_


#Compare Actual and Predicted Data
comparison = pd.DataFrame({'Actual':y_Test, 'Predicted':np.round(y_Pred)})
print(comparison)

# Predicting Future; y=mx+c
y_Pred_12_years = (coef*12)+intercept
print('Salary for 12 years experience = ',y_Pred_12_years)



bias = regressor.score(X_Train, y_Train)
print('Bias Percent = ',bias)


variance = regressor.score(X_Test, y_Test)
print('Variance = ',variance)

prediction_score = regressor.score(X_Test, y_Pred)
print('Prediction Score = ',variance)

print("Mean of Dataset = ",salary.mean())
print("Standard Deviation of Dataset = ",salary.std())
print("Variance of Dataset = ",salary.var())
print("Correlation of Dataset = ",salary.corr())
print("Standard Error over Mean of Dataset = ",salary.sem())
print("Skewness of Dataset = ",salary.skew())

import scipy.stats as stats
salary.apply(stats.zscore)

#SSR
y_mean = np.mean(y)
SSR = np.sum((y_Pred - y_mean)**2)
print('SSR = ',SSR)

#SSE
y =y[0:6]
SSE = np.sum((y-y_Pred)**2)
print('SSE = ',SSE)

# SST 
mean_total = np.mean(salary.values) # here df.to_numpy()will convert pandas Dataframe to Nump
SST = np.sum((salary.values-mean_total)**2)
print("SST = ",SST)


# R Square
r_square = 1 - (SSR/SST)
print("R Score",r_square)
