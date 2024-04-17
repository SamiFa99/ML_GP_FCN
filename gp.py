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
from train_2L_FCN import *


def forward_pass(Theta, x):
    """
    Computes the forward pass through the network.
        :param x:
        :param Theta:
    Returns:
        np.ndarray: Output of the network
    """
    w, a, bias2 = Theta

    z_1 = np.matmul(np.transpose(x), np.transpose(w))
    a_1 = np.square(z_1)
    z_2 = np.dot(a_1, a) + bias2

    return z_2, a_1


def kernel_function(Theta, x_1, x_2):

    k = np.mean(forward_pass(Theta, x_1)[0] * forward_pass(Theta, x_2)[0])
    return k


def K(Theta, X):

    K = np.zeros((X.shape[0], X.shape[0]))
    for r in range(X.shape[0]):
        for c in range(X.shape[0]):
            K[r, c] = kernel_function(Theta, X[r], X[c])
    return K


def f_prediction(Theta, X_test, X, y_train, temperature):

    many_f_mean = []
    cov_mat = np.linalg.inv(K(Theta, X) + np.identity(X.shape[0]) * 1 / temperature)  # matrix to invert
    for x in X_test:
        f_mean = 0
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                f_mean += kernel_function(Theta, x, X[i]) * cov_mat[i, j] * y_train[j]
        many_f_mean.append(f_mean)

    return many_f_mean



def main():
    """
    The main function that demonstrates the usage of the implemented Gaussian Process Regression method.
    """
    # Define the network dimensions and hyperparameters
    input_size = 3
    hidden_size = 400
    output_size = 10
    temperature = 0.1
    n = 40

    # Create data point
    X_train = np.random.randint(-4, 4, (n, 3))  # Generate n random points in the XYZ plane
    # Calculate the squared distance from the origin for each point
    y_train = np.sqrt(np.sum(X_train ** 2, axis=1, keepdims=True))

    # Create test point
    X_test = np.random.randint(-10, 10, (n, 3))  # Generate n random points in the XYZ plane
    # Calculate the squared distance from the origin for each point
    y_test = np.sqrt(np.sum(X_train ** 2, axis=1, keepdims=True))

    # Initialize the neural network model
    a = np.random.normal(0, 1. / np.sqrt(hidden_size), (hidden_size, 1))
    w = np.random.normal(0, 1. / np.sqrt(input_size), (hidden_size, input_size))
    bias2 = np.random.normal(0, 1. / np.sqrt(input_size), (output_size, 1))
    Theta = (w, a, bias2)

    # Gaussian process predictor
    f_mean_prediction = f_prediction(Theta, X_test, X_train, y_train, temperature)
    average_loss = sum((f_mean_prediction - y_test) ** 2) / len(y_test)
    print("avrage f_mean_prediction: ", sum(f_mean_prediction) / len(f_mean_prediction))
    print(f"average loss for test points: {average_loss}")

    # # Calculate the marginal prior distribution, f(x_1)
    # his1, bins1 = np.histogram(f_mean_prediction, bins=10**5 // 400, density=True)
    #
    # # Plot
    # plt.figure()
    # plt.plot(bins1[:-1], f_mean_prediction, '.')
    # plt.legend()
    # plt.ylabel('P(f)')
    # plt.xlabel('f')
    # plt.show()


if __name__ == "__main__":  # TODO: add several test points and show distribution
    main()
