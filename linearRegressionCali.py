import numpy as np
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time

def find_coefficients(X, y):
    # Add a column of ones to X for the intercept term
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    
    # Compute the matrix (X^T X) and the vector (X^T y)
    X_transpose = X.T
    X_transpose_X = np.dot(X_transpose, X)
    X_transpose_y = np.dot(X_transpose, y)
    
    # Compute the coefficients using the Normal Equation
    # np.linalg.inv calculates the matrix inverse
    # np.dot performs matrix multiplication
    coefficients = np.dot(np.linalg.inv(X_transpose_X), X_transpose_y)
    
    return coefficients

def calculate_mse(y_true, y_pred):
    # Calculate the Mean Squared Error between actual and predicted values
    mse = np.mean((y_true - y_pred) ** 2)
    return mse


def predict_values(X, coefficients):
    # Add a column of ones to X for the intercept term
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    
    # Calculate predictions: dot product of X and coefficients
    predictions = np.dot(X, coefficients)
    return predictions


# Example usage with previously defined variables
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

start = time.time()
# Load the data
data = fetch_california_housing()
X = data.data
y = data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute the coefficients using the training data
coefficients = find_coefficients(X_train, y_train)

# Predict using the test data
test_predictions = predict_values(X_test, coefficients)

# Calculate and print the MSE for the test predictions
test_mse = calculate_mse(y_test, test_predictions)
print("Mean Squared Error on Test Data:", test_mse)

end = time.time()
print("total time: "+str(end-start))
