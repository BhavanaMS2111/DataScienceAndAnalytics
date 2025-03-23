# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 23:12:52 2025

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


goldPricesDataset = pd.read_csv(r'D:\Data Science and Machine Learning\GoldPrice_Prediction\gold_prices.csv')

#Independant Variable
X = goldPricesDataset.iloc[:,0]
#Dependant Variable
y = goldPricesDataset.iloc[:,2]

from sklearn.model_selection import train_test_split
X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,y,test_size=.2,random_state=0)

X_Train = X_Train.values.reshape(-1,1)
X_Test = X_Test.values.reshape(-1,1)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)
Y_Pred = regressor.predict(X_Test)

print("Coefficient of Variance of Data (m) = ",regressor.coef_)
print("Incercept (c) = ",regressor.intercept_)


# Visualizations in Dataset with Training Data
plt.scatter(X_Train, Y_Train, color='red')  # Plot actual training data points
plt.plot(X_Train, regressor.predict(X_Train), color='blue')  # Plot predicted line using training data
plt.title('Visualization for Training Data Set')
plt.xlabel('Opening Price of Gold')
plt.ylabel('Closing Price of Gold')
plt.show()

# Visualizations in Dataset with Test Data
plt.scatter(X_Test, Y_Test, color='red')  # Plot actual test data points
plt.plot(X_Train, regressor.predict(X_Train), color='blue')  # Use training data line to visualize
plt.title('Visualization for Test Data Set')
plt.xlabel('Opening Price of Gold')
plt.ylabel('Closing Price of Gold')
plt.show()

# Pass today's today_conversion_rate and get the Closing Price
today_conversion_rate = 85.990
# y = mx + c
model_predicted = regressor.coef_ * today_conversion_rate + regressor.intercept_
print("Price Predicted by Model = ",model_predicted)
print('Actual Price = 8428')

df = pd.DataFrame({'Actual Rate':8428,'Model Predicted':model_predicted})
print('\n\n\n',df)

# Model Analysis
bias = regressor.score(X_Train, Y_Train)
variance = regressor.score(X_Test, Y_Test)

metrics = pd.DataFrame({'Metric': ['Bias', 'Variance'], 'Value': [bias, variance]})

print('\n\n\n',metrics)


# Sum of Squares Regression (SSR)
SSR = np.sum((Y_Pred - np.mean(Y_Test)) ** 2)
# Sum of Squares Error (SSE)
SSE = np.sum((Y_Test - Y_Pred) ** 2)
# Total Sum of Squares (SST)
SST = np.sum((y - np.mean(y)) ** 2)

# Create a DataFrame to show SSR, SSE, and SST
sum_of_squares = pd.DataFrame({'Measure': ['SSR', 'SSE', 'SST'], 'Value': [SSR, SSE, SST]})

print('\n\nSum of Squares Analysis:')
print(sum_of_squares)


R_Square = 1- (SSR/SST)
print('\n\n\n R Square = ',R_Square)

