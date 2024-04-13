import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class FullyConnectedNetwork:
    """
    A simple two-layer fully connected neural network with quadratic activation
    in the hidden layer and ReLU activation in the output layer, suitable for regression
    or binary classification tasks.

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

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Applies the sigmoid activation function using a safe method to prevent overflow."""
        # Clipping input values to avoid very large values in exp()
        z = np.clip(z, -10, 10)
        return 1 / (1 + np.exp(-z))

    def forward_pass(self, x: np.ndarray):
        """
        Computes the forward pass through the network.

        Args:
            x (np.ndarray): Input tensor.

        Returns:
            np.ndarray: Output of the network.
        """
        z_1 = np.dot(x, self.weights1) + self.biases1
        a_1 = self.quadratic_activation(z_1)
        z_2 = np.dot(a_1, self.weights2) + self.biases2
        output = self.sigmoid(z_2)
        return output, a_1, z_2

    @staticmethod
    def compute_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
        epsilon = 1e-12
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

    def backpropagation(self, x: np.ndarray, a_1: np.ndarray, z_2: np.ndarray,
                        predictions: np.ndarray, targets: np.ndarray):
        """
        Computes gradients and updates parameters using Langevin dynamics.
        """

        # Compute gradients
        grad_weights1, grad_biases1, grad_weights2, grad_biases2 = self.compute_gradients(x, a_1, z_2, predictions,
                                                                                          targets)

        # Langevin dynamics update for weights and biases with decay
        self.weights1 -= (self.learning_rate * grad_weights1 + self.gamma * self.weights1 +
                          np.sqrt(2 * self.learning_rate * self.temperature) * np.random.randn(*self.weights1.shape))
        self.biases1 -= (self.learning_rate * grad_biases1 + self.gamma * self.biases1 +
                         np.sqrt(2 * self.learning_rate * self.temperature) * np.random.randn(*self.biases1.shape))

        self.weights2 -= (self.learning_rate * grad_weights2 + self.gamma * self.weights2 +
                          np.sqrt(2 * self.learning_rate * self.temperature) * np.random.randn(*self.weights2.shape))
        self.biases2 -= (self.learning_rate * grad_biases2 + self.gamma * self.biases2 +
                         np.sqrt(2 * self.learning_rate * self.temperature) * np.random.randn(*self.biases2.shape))

    def compute_gradients(self, x: np.ndarray, a_1: np.ndarray, z_2: np.ndarray,
                          predictions: np.ndarray, targets: np.ndarray):
        N = targets.shape[0]  # Number of samples in the batch

        # Ensure predictions and targets are correctly shaped
        d_loss_d_predictions = predictions - targets  # should be (batch_size, 1)

        # Gradients for the output layer
        d_loss_d_z2 = d_loss_d_predictions  # should be (batch_size, 1)
        d_loss_d_weights2 = np.dot(a_1.T, d_loss_d_z2) / N
        d_loss_d_biases2 = np.sum(d_loss_d_z2, axis=0) / N

        # Backpropagation to the hidden layer
        d_z2_d_a1 = self.weights2  # should be (5, 1)
        d_loss_d_a1 = np.dot(d_loss_d_z2, d_z2_d_a1.T)

        # Derivative computation for quadratic activation
        z_1 = np.dot(x, self.weights1) + self.biases1
        d_a1_d_z1 = 2 * z_1
        d_loss_d_z1 = d_loss_d_a1 * d_a1_d_z1
        d_loss_d_weights1 = np.dot(x.T, d_loss_d_z1) / N
        d_loss_d_biases1 = np.sum(d_loss_d_z1, axis=0) / N

        return d_loss_d_weights1, d_loss_d_biases1, d_loss_d_weights2, d_loss_d_biases2

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the class labels or values for the input data.

        Args:
            x (np.ndarray): Input tensor.

        Returns:
            np.ndarray: Predicted class labels or values.
        """

        output, _, _ = self.forward_pass(x)  # Unpack the tuple to get the output
        predicted_labels = (output > 0.5).astype(int)  # Apply threshold
        return predicted_labels

    def evaluate_accuracy(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluates the model's accuracy on the provided test dataset.

        Args:
            x_test (np.ndarray): Test input tensor.
            y_test (np.ndarray): Test target labels or values.

        Returns:
            float: The model's accuracy on the test data.
        """

        predicted_labels = self.predict(x_test)
        accuracy = np.mean(predicted_labels == y_test)  # Calculating the accuracy
        return accuracy


def load_data(filename):
    """
    Loads and preprocesses the kidney stone prediction dataset to randomly select
    40 samples for training and the remaining for testing.

    Args:
        filename (str): The path to the dataset file.

    Returns:
        tuple: A tuple containing the training and testing datasets:
               (X_train, X_test, y_train, y_test).
    """
    # Load the dataset
    df = pd.read_csv(filename)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # 'target' is the name of the column indicating the presence of kidney stones
    X = df.drop('target', axis=1).values
    y = df['target'].values

    # Randomly select 40 samples for training; the rest will be for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=40, random_state=42)

    # Normalize the feature data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape y_train and y_test to ensure they are 2-dimensional (n_samples, 1)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return X_train, X_test, y_train, y_test


def train_model(model, x_train, y_train, batch_size=10, tolerance=0.001, max_iterations=1000):
    """
    Trains the model on the given dataset using mini-batches and stops when the loss change is below a tolerance level.

    Args:
        model (FullyConnectedNetwork): The neural network model to be trained.
        x_train (np.ndarray): Training input data.
        y_train (np.ndarray): Training target labels.
        batch_size (int): Number of samples per batch.
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
        x_train = x_train[indices]
        y_train = y_train[indices]

        total_loss = 0
        batches = len(x_train) // batch_size

        for i in range(batches):
            start = i * batch_size
            end = start + batch_size
            x_batch = x_train[start:end]
            y_batch = y_train[start:end]

            predictions, a_1, z_2 = model.forward_pass(x_batch)
            loss = model.compute_loss(predictions, y_batch)
            total_loss += loss

            model.backpropagation(x_batch, a_1, z_2, predictions, y_batch)

        # Compute the average loss for this iteration
        average_loss = total_loss / batches
        loss_history.append(average_loss)

        # Check for convergence
        if np.abs(last_loss - average_loss) < tolerance:
            print(f"Convergence reached after {iteration + 1} iterations.")
            break

        last_loss = average_loss
        iteration += 1

        # Optionally print the average loss every few iterations
        if iteration % 10 == 0 or iteration == 1:
            print(f'Iteration {iteration}, Loss: {average_loss:.4f}')

    # Plot the loss history after training
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss During Training')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """
    Main function to execute the training and evaluation.
    """
    file = "kindey stone urine analysis.csv"  # Ensure the filename is spelled correctly

    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_data(file)

    # Define the network dimensions and hyperparameters
    input_size = X_train.shape[1]
    hidden_size = 5
    output_size = 1
    learning_rate = 0.01
    temperature = 1
    gamma = 0  # Choose an appropriate value for gamma

    # Initialize the neural network model with the new gamma parameter
    model = FullyConnectedNetwork(input_size, hidden_size, output_size, learning_rate, temperature, gamma)

    # Train the model with batch processing
    print("Starting training...")
    train_model(model, X_train, y_train, batch_size=5, tolerance=0.001, max_iterations=500)
    print("Training completed.")

    # Evaluate the model's performance on the test set
    accuracy = model.evaluate_accuracy(X_test, y_test)
    print(f"Model accuracy on test set: {accuracy:.2%}")


if __name__ == "__main__":
    main()
