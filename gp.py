 numpy as np
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
        self.map_mat = self.map_matrix()
        self.K_matrix = self.K_matrix()
        self.A = np.transpose(self.K_matrix + np.identity(self.K_matrix.size) * self.noize_variance)
        self.k_vector = self.k_vector()

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

    def map_matrix(self):
        """
        Build a new matrix which is an aggregation of columns map_func(x) for all cases in the training set.
        We have a training set of n observations. The column vector inputs for all n cases are aggregated in
        the D × n design matrix.
        :return: a new matrix which is an aggregation of columns map_func(x) for all cases in the training set.
        """

        mat = np.array([])
        for inp_vec in np.transpose(self.X_train):  # go through input vectors in dataset
            np.append(mat, self.map_func(inp_vec))
        return np.transpose(mat)

    def K_matrix(self):
        """
        create covariances matrix K.
        :return: K
        """
        map_mat = self.map_matrix()
        K = np.matmul(np.matmul(np.transpose(map_mat), self.prior_cov_mat), map_mat)
        return K

    def kernel_func(self, x, y):
        """
        :param x:
        :param y:

        :return: int
        """
        k = np.matmul(np.matmul(np.transpose(self.map_func(x)), self.prior_cov_mat), self.map_func(y))
        return k

    def k_vector(self):
        vec = []
        for i in range(self.X_train.shape[1]):
            vec.append(self.kernel_func(self.X_train.shape[:, i], self.x_test))
        return np.array(vec)

    def mean_f_star(self, k_vector):
        """
        :return: predictive distribution mean
        """
        coefficients = np.matmul(np.matmul(self.A, self.y_train))
        f_star_mean = sum([coefficients[i] * k_vector[i] for i in range(len(self.k_vector))])
        return f_star_mean

    def var_f_star(self):
        """
        :return: predictive distribution variance
        """
        f_star_var = self.kernel_func(self.x_test, self.x_test) - np.matmul(np.matmul(np.transpose(self.k_vector),
                                                                                      self.A), self.k_vector)
        return f_star_var


if __name__ == '__main__':
    a = np.random.rand(4)
