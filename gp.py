import numpy as np
import matplotlib.pyplot as plt


class f_star:
    """
    Predictive distribution
    """
    def __init__(self, X_train, y_train, x_test, noize_variance, prior_cov_mat):
        """
        :param X_train:
        :param y_train:
        :param x_test:
        :param noize_variance:
        """
        self.prior_cov_mat = prior_cov_mat
        self.noize_variance = noize_variance
        self.x_test = x_test
        self.y_train = y_train
        self.X_train = X_train
        self.map_mat = map_matrix(X_train)
        self.K_matrix = self.K_matrix()
        self.A = np.transpose(self.K_matrix + np.identity(self.K_matrix.size) * self.noize_variance)
        self.k_vector = k_vector()

    @staticmethod
    def map_func(vector):
        """
        mapping a vector [x1, x2, ..., xn] to a higher dim
        :param vector: a vector [x1, x2, ..., xn]
        :return: vector [x1, x2, ..., xN]
        """
        new_v = []
        for i in vector:
            for j in vector:
                if j >= i:
                    new_v.append(i * j)
        return new_v

    @staticmethod
    def map_matrix(X):
        """
        Build a new matrix which is an aggregation of columns map_func(x) for all cases in the training set.
        We have a training set of n observations. The column vector inputs for all n cases are aggregated in
        the D × n design matrix.
        :param X: D × n design matrix. Aggregation of column vector inputs for all cases in the training set.
        :return: a new matrix which is an aggregation of columns map_func(x) for all cases in the training set.
        """

        mat = np.array([])
        for inp_vec in np.transpose(X):  # go through input vectors in dataset
            mat.append(map_func(inp_vec))
        return np.transpose(mat)

    def K_matrix(self):
        """
        create covariances matrix K.
        :return: K
        """
        map_mat = map_matrix(self.X)
        K = np.matmul(np.matmul(np.transpose(map_mat), self.prior_cov_mat), map_mat)
        return K

    def kernel_func(self, x, y):
        """
        :param x:
        :param y:

        :return: int
        """
        k = np.matmul(np.matmul(np.transpose(map_func(x)), self.prior_cov_mat), map_func(y))
        return k

    @staticmethod
    def k_vector():
        vec = []
        for i in range(X.shape[1]):
            vec.append(kernel_func(X[:, i], x_test))
        return np.array(vec)

    def mean_f_star(self, k_vector):
        """
        :return: predictive distribution mean
        """
        coefficients = np.matmul(np.matmul(self.A, y_train))
        f_star_mean = sum([coefficients[i] * k_vector[i] for i in range(len(self.k_vector))])
        return f_star_mean

    def var_f_star(self):
        """
        :return: predictive distribution variance
        """
        f_star_var = kernel_func(x_test, x_test) - np.matmul(np.matmul(np.transpose(self.k_vector), A), self.k_vector)
        return f_star_var


if __name__ == '__main__':
    a = np.random.rand(4)
    print(map_func(a))