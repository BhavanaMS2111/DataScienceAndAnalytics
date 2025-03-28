# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 11:30:44 2025

@author: BHavana
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Load the dataset
houseData = pd.read_csv(r'D:\Data Science and Machine Learning\March_Bhav\26th_MultipleLinearRegression\HousingPrices\Enhanced_House_Data_Bangalore.csv')

# Preview the data
print(houseData.head())

# Check for null values
print('Check Null Values:')
print(houseData.isnull().sum())

# Drop the Irrelevant Columns
houseData = houseData.drop(['id', 'date'], axis=1)

# Encode categorical columns: 'metro_proximity', 'security_level' using Label Encoding
labelEncoder = LabelEncoder()
for col in ['metro_proximity', 'security_level']:
    houseData[col] = labelEncoder.fit_transform(houseData[col])

# One-hot encode 'builder_name' for better accuracy
houseData = pd.get_dummies(houseData, columns=['builder_name'], drop_first=True)

# Define independent (X) and dependent (Y) variables
X = houseData.drop('price', axis=1).values  # All columns except 'price'
Y = houseData['price'].values  # Target variable 'price'

# Check for NaN values properly
print(f"Any NaN in X? {pd.isnull(X).sum().sum()}")
print(f"Any NaN in Y? {pd.isnull(Y).sum()}")

# Split the data into training and test sets (70% training, 30% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Create and train the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predict values using X_test
Y_pred = regressor.predict(X_test)

# Print model coefficients and intercept
# Create a summary dictionary with rounded values
model_summary = {
    "Intercept": round(regressor.intercept_, 3),
    "Coefficients": [round(coeff, 3) for coeff in regressor.coef_]
}
print(model_summary)

# Plot Actual vs Predicted Prices
plt.figure(figsize=(8, 6))
plt.scatter(Y_test, Y_pred, color='blue', label='Actual vs Predicted')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--')
plt.xlabel('Actual Price (₹)')
plt.ylabel('Predicted Price (₹)')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.show()

# Feature importance visualization (coefficients)
feature_names = houseData.drop('price', axis=1).columns
coefficients = regressor.coef_
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
coef_df = coef_df.sort_values(by='Coefficient', ascending=False).head(10)

# Plot top 10 features with highest impact
plt.figure(figsize=(12, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='viridis')
plt.title('Top 10 Features Influencing House Price')
plt.xlabel('Coefficient Value (Impact on Price)')
plt.ylabel('Feature')
plt.show()

#Backward Elimination
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((21613,19)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(Y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17]]
X_Modeled = backwardElimination(X_opt, SL)
