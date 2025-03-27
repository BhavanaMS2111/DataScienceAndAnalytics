import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Configurable File Paths
DATA_PATH = r'D:\Data Science and Machine Learning\GoldPrice_Prediction\gold_prices.csv'
PICKLE_PATH = r'D:\Data Science and Machine Learning\GoldPrice_Prediction\gold_price_predictor.pkl'

# Load Dataset with Error Handling
try:
    goldPricesDataset = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: The file {DATA_PATH} was not found.")
    exit()

#Independant Variable
X = goldPricesDataset.iloc[:,0]
#Dependant Variable
y = goldPricesDataset.iloc[:,2]


# Split Dataset
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size=0.2, random_state=0)

X_Train = X_Train.values.reshape(-1, 1)
X_Test = X_Test.values.reshape(-1, 1)

# Model Training and Prediction
regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)
Y_Pred = regressor.predict(X_Test)

print("Coefficient of Variance of Data (m) = ", regressor.coef_)
print("Intercept (c) = ", regressor.intercept_)

# Visualization Function
def plot_visualization(X, Y, title, X_label, Y_label):
    plt.scatter(X, Y, color='red')
    plt.plot(X, regressor.predict(X), color='blue')
    plt.title(title)
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.show()

# Visualize Training and Test Data
plot_visualization(X_Train, Y_Train, 'Visualization for Training Data Set', 'Opening Price of Gold', 'Closing Price of Gold')
plot_visualization(X_Test, Y_Test, 'Visualization for Test Data Set', 'Opening Price of Gold', 'Closing Price of Gold')

# Model Performance Metrics
bias = regressor.score(X_Train, Y_Train)
variance = regressor.score(X_Test, Y_Test)
train_mse = mean_squared_error(Y_Train, regressor.predict(X_Train))
test_mse = mean_squared_error(Y_Test, Y_Pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

# Summary of Model Performance
summary = pd.DataFrame({
    'Metric': ['R^2 (Train)', 'R^2 (Test)', 'Train RMSE', 'Test RMSE'],
    'Value': [bias, variance, train_rmse, test_rmse]
})
print(summary)

# Save Model to Disk
try:
    with open(PICKLE_PATH, 'wb') as file:
        pickle.dump(regressor, file)
    print(f"Model has been pickled and saved as {PICKLE_PATH}")
except Exception as e:
    print(f"Error while saving model: {e}")

# Predicted Price for Fixed Values
y_100 = regressor.predict([[100]])
y_200 = regressor.predict([[200]])
print(f"Predicted Gold Rate for Rs. 100 Conversion Rate: INR{y_100[0]:,.2f}")
print(f"Predicted Gold Rate for Rs. 200 Conversion Rate: INR{y_200[0]:,.2f}")
