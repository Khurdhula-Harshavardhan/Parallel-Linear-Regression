import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

def fit_linear_regression(X, y):
    """
    Fit a linear regression model using the normal equation.
    X: Feature matrix (numpy array)
    y: Target vector (numpy array)
    """
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add intercept term (x0 = 1)
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta

def predict(X, theta):
    """
    Make predictions using the linear regression model.
    X: Feature matrix (numpy array)
    theta: Coefficients vector (numpy array)
    """
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add intercept term
    return X_b.dot(theta)

def mean_squared_error(y_true, y_pred):
    """
    Calculate the mean squared error of the model.
    y_true: Actual values (numpy array)
    y_pred: Predicted values (numpy array)
    """
    return np.mean((y_true - y_pred) ** 2)

# Load your data
data = pd.read_csv('dataset/yo.csv')

# Prepare the data
X = data.drop('price', axis=1).values
y = (data['price'].values)

# Fit the model
theta = fit_linear_regression(X, y)

# Make predictions (using X to demonstrate; typically, you'd use unseen data)
y_pred = predict(X, theta)

# Calculate MSE
error = mean_squared_error(y, y_pred)

print('Model coefficients:', theta)
print('Mean Squared Error:', error)
print("R2_score: %f"%(r2_score(y, y_pred)))
