import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import multiprocessing
from sklearn.datasets import fetch_california_housing
from functools import partial
import time

def compute_xtx(chunk_indices, X):
    return X[chunk_indices].T @ X[chunk_indices]

def compute_xty(chunk_indices, X, y):
    return X[chunk_indices].T @ y[chunk_indices]

def parallel_linear_regression(X_train, y_train, num_processes=None):
    X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    num_processes = num_processes or multiprocessing.cpu_count()
    chunk_sizes = np.array_split(range(X_b.shape[0]), num_processes)
    pool = multiprocessing.Pool(processes=num_processes)
    partial_xtx = partial(compute_xtx, X=X_b)
    xtx_parts = pool.map(partial_xtx, chunk_sizes)
    XTX = sum(xtx_parts)
    partial_xty = partial(compute_xty, X=X_b, y=y_train)
    xty_parts = pool.map(partial_xty, chunk_sizes)
    XTy = sum(xty_parts)
    pool.close()
    pool.join()
    theta_best = np.linalg.inv(XTX).dot(XTy)
    return theta_best

def main():
    # Load your data here (example with dummy data)
    data = fetch_california_housing()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    coefficients = parallel_linear_regression(X_train, y_train)
    start = time.time()
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    
    y_pred = X_test_b.dot(coefficients)
    mse = mean_squared_error(y_test, y_pred)
    end = time.time()
    print("Total time taken: "+str(end - start))
    print("Coefficients:", coefficients)
    print("Mean Squared Error:", mse)
    
    
if __name__ == '__main__':
    multiprocessing.freeze_support()  # This is important for Windows
    main()
