import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class FullyConnectedNetwork:
    """
    A simple two-layer fully connected neural network with quadratic activation
    in the hidden layer

    Attributes:
        input_size (int): Size of the input layer.
        hidden_size (int): Number of neurons in the hidden layer.
        output_size (int): Size of the output layer.
        learning_rate (float): Learning rate for the optimizer.
        temperature (float): Temperature parameter for Langevin dynamics.
        weights1 (ndarray): Weights for the first layer.
        biases1 (ndarray): Biases for the first layer.
        weights2 (ndarray): Weights for the second layer.
        biases2 (ndarray): Biases for the second layer.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 learning_rate: float, temperature: float, gamma: float):
        """
        Initializes the network with specified sizes and learning parameters.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.gamma = gamma
        # Initialize weights with a smaller standard deviation
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.biases1 = np.zeros(hidden_size)
        self.biases2 = np.zeros(output_size)

    @staticmethod
    def quadratic_activation(x: np.ndarray) -> np.ndarray:
        """Applies a quadratic activation function to the input tensor."""
        return x ** 2

    def forward_pass(self, x: np.ndarray):
        """
        Computes the forward pass through the network.

        Args:
            x (np.ndarray): Input tensor.

        Returns:
            np.ndarray: Output of the network.
        """
        z_1 = np.dot(x, self.weights1) + self.biases1
        a_1 = self.quadratic_activation(z_1) - (1 / 3) * np.ones((len(z_1), 1))
        z_2 = np.dot(a_1, self.weights2) + self.biases2
        return z_2, a_1

    @staticmethod
    def compute_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Computes the mean squared error loss.

        Args:
            predictions (np.ndarray): Output predictions from the network.
            targets (np.ndarray): Actual target values.

        Returns:
            float: The computed mean squared error loss.
        """
        return np.mean((predictions - targets) ** 2)

    def backpropagation(self, x: np.ndarray, a_1: np.ndarray,
                        predictions: np.ndarray, targets: np.ndarray):
        """
        Computes gradients and updates parameters using Langevin dynamics.
        """

        # Compute gradients
        grad_weights1, grad_biases1, grad_weights2, grad_biases2 = self.compute_gradients(x, a_1, predictions,
                                                                                          targets)

        # Langevin dynamics update for weights and biases with decay
        # For weights
        self.weights1 -= self.learning_rate * grad_weights1 + (self.gamma * self.weights1) + np.sqrt(
            2 * self.learning_rate * self.temperature) * np.random.randn(*self.weights1.shape)
        self.weights2 -= self.learning_rate * grad_weights2 + (self.gamma * self.weights2) + np.sqrt(
            2 * self.learning_rate * self.temperature) * np.random.randn(*self.weights2.shape)

        # For biases (note that typically biases are not decayed, but adding noise is still valid)
        # self.biases1 -= self.learning_rate * grad_biases1 + np.sqrt(
        #    2 * self.learning_rate * self.temperature) * np.random.randn(*self.biases1.shape)
        self.biases2 -= self.learning_rate * grad_biases2 + np.sqrt(
            2 * self.learning_rate * self.temperature) * np.random.randn(*self.biases2.shape)

    def compute_gradients(self, x: np.ndarray, a_1: np.ndarray, predictions: np.ndarray, targets: np.ndarray):
        """
        Computes gradients for backpropagation using MSE loss.

        Args:
            x (np.ndarray): Input data.
            a_1 (np.ndarray): Activations from the hidden layer.
            predictions (np.ndarray): Network output after activation.
            targets (np.ndarray): True target values.

        Returns:
            tuple: Gradients for weights and biases in both layers.
        """
        N = targets.shape[0]  # Number of samples in the batch

        # Gradient of loss w.r.t. output predictions
        d_loss_d_predictions = 2 * (predictions - targets) / N

        # Gradients for the output layer
        d_loss_d_z2 = d_loss_d_predictions
        d_loss_d_weights2 = np.dot(a_1.T, d_loss_d_z2)
        d_loss_d_biases2 = np.sum(d_loss_d_z2, axis=0)

        # Backpropagation to the hidden layer
        d_z2_d_a1 = self.weights2
        d_loss_d_a1 = np.dot(d_loss_d_z2, d_z2_d_a1.T)

        # Derivative computation for quadratic activation
        z_1 = np.dot(x, self.weights1) + self.biases1
        d_a1_d_z1 = 2 * z_1  # derivative of x^2 is 2x
        d_loss_d_z1 = d_loss_d_a1 * d_a1_d_z1
        d_loss_d_weights1 = np.dot(x.T, d_loss_d_z1)
        d_loss_d_biases1 = np.sum(d_loss_d_z1, axis=0)

        return d_loss_d_weights1, d_loss_d_biases1, d_loss_d_weights2, d_loss_d_biases2

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the squared distances for the input data based on the model's computation.

        Args:
            x (np.ndarray): Input tensor.

        Returns:
            np.ndarray: Predicted squared distances.
        """
        output, _ = self.forward_pass(x)  # Unpack the tuple to get the output
        return output

    def evaluate_accuracy(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluates the model's performance using Mean Squared Error (MSE) as the metric.

        Args:
            x_test (np.ndarray): Test input tensor.
            y_test (np.ndarray): Test target squared distances.

        Returns:
            float: The MSE between predicted and actual squared distances.
        """
        predictions = self.predict(x_test)
        mse = np.mean((predictions - y_test) ** 2)  # Calculate the mean squared error
        return mse


def load_data(n):
    """
    Generates n random points in the XYZ plane and calculates the squared distance from the origin for each point.

    Args:
        n (int): Number of samples to generate.

    Returns:
        tuple: A tuple containing the training datasets:
               (X_train, y_train).
    """
    # Generate n random points in the XYZ plane
    X_train = np.random.rand(n, 3)  # random values between 0 and 1 for 3 dimensions

    # Calculate the squared distance from the origin for each point
    y_train = np.sum(X_train ** 2, axis=1, keepdims=True)

    # Normalize the feature data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    return X_train, y_train


def train_model(model, x_train, y_train, tolerance=0.0001, max_iterations=10000):
    """
    Trains the model on the given dataset using stochastic gradient descent and stops when the loss change is below a
    tolerance level.

    Args:
        model (FullyConnectedNetwork): The neural network model to be trained.
        x_train (np.ndarray): Training input data.
        y_train (np.ndarray): Training target labels.
        tolerance (float): Minimum change in loss required to continue training.
        max_iterations (int): Maximum number of iterations to prevent infinite loops.

    """
    # Initialize variables for tracking the loss and iterations
    loss_history = []
    iteration = 0
    last_loss = float('inf')

    while iteration < max_iterations:
        # Shuffle the data at the beginning of each epoch to prevent cycles.
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]

        total_loss = 0
        for i in range(len(x_train_shuffled)):
            x_sample = x_train_shuffled[i:i + 1]  # Get sample by sample
            y_sample = y_train_shuffled[i:i + 1]

            # Perform forward pass on the single sample
            predictions, a_1 = model.forward_pass(x_sample)
            loss = model.compute_loss(predictions, y_sample)
            total_loss += loss

            # Perform backpropagation on the single sample
            model.backpropagation(x_sample, a_1, predictions, y_sample)

        # Compute the average loss for this iteration
        average_loss = total_loss / len(y_train_shuffled)
        loss_history.append(average_loss)

        # Optionally print the average loss every few iterations
        if iteration % 10 == 0 or iteration == 1:
            print(f'Iteration {iteration + 1}, Average Loss: {average_loss:.4f}')

        # Check for convergence
        if abs(average_loss) < tolerance:
            print(f"Convergence reached after {iteration + 1} iterations.")
            break

        last_loss = average_loss
        iteration += 1

    # Plot the loss history after training
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Average Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Average Loss')
    plt.title('Average Loss During Training')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """
    Main function to execute the training and evaluation.
    """

    # Load and preprocess the data
    X_train, y_train, = load_data(20)

    # Define the network dimensions and hyperparameters
    input_size = 3
    hidden_size = 30
    output_size = 1
    learning_rate = 0.00001
    temperature = 1
    lamdbda = 1  # Choose an appropriate value for gamma

    # Initialize the neural network model with the new gamma parameter
    model = FullyConnectedNetwork(input_size, hidden_size, output_size, learning_rate, temperature, lamdbda)

    # Train the model with batch processing
    print("Starting training...")
    train_model(model, X_train, y_train, tolerance=0.1, max_iterations=1000)
    print("Training completed.")
    x_test, y_test = load_data(1)  # test points
    # Evaluate the model's performance on the test set
    loss = model.evaluate_accuracy(x_test, y_test)
    print(f"average loss for test points: {loss}")


if __name__ == "__main__":
    main()
