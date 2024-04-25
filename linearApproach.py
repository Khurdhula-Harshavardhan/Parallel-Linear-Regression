import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import time


# Load California housing dataset
data = fetch_california_housing()
X = data.data
print(X.shape)
y = data.target

# Split data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add intercept term
X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

start = time.time()

# Calculate coefficients using the Normal Equation
theta_best = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)

# Make predictions on the testing set
y_pred = X_test_b.dot(theta_best)

end = time.time()
print("Total time taken: "+str(end - start))
# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Coefficients:", theta_best)
print("Mean Squared Error:", mse)
r2 =r2_score(y_test, y_pred)
