import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

salary = pd.read_csv(r'D:\Data Science and Machine Learning\March\18thMachineLearning\Salary_data.csv')

X = salary.iloc[:,0]

y = salary.iloc[:,-1]

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

y_12 = regressor.predict([[12]])
y_20 = regressor.predict([[20]])
print(f"Predicted salary for 12 years of experience: ${y_12[0]:,.2f}")
print(f"Predicted salary for 20 years of experience: ${y_20[0]:,.2f}")

# Check model performance
bias = regressor.score(X_Train, y_Train)
variance = regressor.score(X_Test, y_Test)
train_mse = mean_squared_error(y_Train, regressor.predict(X_Train))
test_mse = mean_squared_error(y_Test, y_Pred)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

# Save the trained model to disk
import pickle
filename = 'linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as linear_regression_model.pkl") 

import os
print(os.getcwd())
