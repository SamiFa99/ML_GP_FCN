"""
Kernel-based Gaussian Process Regression using a Fully Connected Neural Network

This script implements Gaussian Process Regression (GPR) using a neural network as the kernel function.
The neural network used in this implementation is a fully connected feedforward neural network with one hidden layer.
The GPR is performed by calculating the kernel function for a test point using the kernel function
defined by the neural network.
The kernet defined as an average of a dot product between two outputs of the network,
with respect to the weights and biases that are drawn from centered Gaussian distribution.

Functions:
    kernel_function(model, f_1, f_2):
        Computes the kernel function value between two output data points using the trained neural network model.

    K(model, X):
        Computes the kernel matrix for a given set of input data points using the trained neural network model.

    f_prediction(model, x_test, X, y):
        Predicts the mean value of the target variable for a test point using Gaussian Process Regression
        with the trained neural network model.

    main():
        The main function that demonstrates the usage of the implemented Gaussian Process Regression method.
"""

import numpy as np
from train_2L_FCN import load_data, FullyConnectedNetwork


def kernel_function(model, f_1, f_2):
    """
    Computes the kernel function value between two input data points using the trained neural network model.

    Args:
        model (FullyConnectedNetwork): The trained neural network model.
        f_1 (ndarray): Input data point x.
        f_2 (ndarray): Input data point y.

    Returns:
        float: The kernel function value between f_1 and f_2.
    """
    k = np.mean(model.forward_pass(f_1)[0] * model.forward_pass(f_2)[0])
    return k


def K(model, X):
    """
    Computes the kernel matrix for a given set of input data points.

    Args:
        model (FullyConnectedNetwork): The untrained neural network model.
        X (ndarray): Input data points matrix.

    Returns:
        ndarray: The kernel matrix computed for the input data points.
    """
    K = np.zeros((X.shape[0], X.shape[0]))
    for r in range(X.shape[0]):
        for c in range(X.shape[0]):
            K[r, c] = kernel_function(model, X[r], X[c])
    return K


def f_prediction(model, x_test, X, y):
    """
    Predicts the mean value of the target variable for a test point using Gaussian Process Regression
    with the untrained neural network model.

    Args:
        model (FullyConnectedNetwork): The untrained neural network model.
        x_test (ndarray): Test data point for which prediction is to be made.
        X (ndarray): Input data points matrix.
        y (ndarray): Target variable values corresponding to the input data points.

    Returns:
        float: The predicted mean value of the target variable for the test point.
    """
    f_mean = 0
    cov_mat = np.linalg.inv(K(model, X) + np.identity(X.shape[0]) * 1 / model.temperature)  # matrix to invert
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            f_mean += kernel_function(model, x_test, X[i]) * cov_mat[i, j] * y[j]
    return f_mean


def main():
    """
    The main function that demonstrates the usage of the implemented Gaussian Process Regression method.
    """
    # Load and preprocess the data
    X_train, y_train, = load_data(20)

    # Define the network dimensions and hyperparameters
    input_size = 3
    hidden_size = 30
    output_size = 1
    learning_rate = 0.00001
    temperature = 1
    Lambda = 1  # a value determining the strength of the penalty (encouraging smaller weights)

    # Initialize the neural network model with the new gamma parameter
    model = FullyConnectedNetwork(input_size, hidden_size, output_size, learning_rate, temperature, Lambda)

    # create test point
    x_test = np.array([0.1, 0.22, 0.3])
    y_test = np.dot(x_test, x_test)

    # Gaussian process predictor
    f_mean_prediction = f_prediction(model, x_test, X_train, y_train)
    err = abs(f_mean_prediction - y_test)
    print("f_mean_prediction: ", f_prediction(model, x_test, X_train, y_train))
    print(f"err for test points: {err}")


if __name__ == "__main__":
    main()
