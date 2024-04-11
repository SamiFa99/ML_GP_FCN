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
                 learning_rate: float, temperature: float):
        """
        Initializes the network with specified sizes and learning parameters.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.z2 = np.zeros(input_size)
        # Initialize weights for the first layer (input to hidden layer)
        # The shape is (input_size, hidden_size) to match the input vector and the size of the hidden layer
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01

        # Initialize biases for the first layer
        # It's a vector of shape (hidden_size,) as there's one bias per neuron in the hidden layer
        self.biases1 = np.zeros(hidden_size)

        # Initialize weights for the second layer (hidden to output layer)
        # The shape is (hidden_size, output_size) to connect the hidden layer to the output layer
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01

        # Initialize biases for the second layer
        # It's a vector of shape (output_size,) as there's one bias per output neuron
        self.biases2 = np.zeros(output_size)

    @staticmethod
    def quadratic_activation(x: np.ndarray) -> np.ndarray:
        """Applies a quadratic activation function to the input tensor."""
        return x ** 2

    @staticmethod
    def relu(z: np.ndarray) -> tuple:
        """Applies the ReLU activation function to the input tensor."""
        return np.maximum(0, z)

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
        output = self.relu(z_2)
        return output, a_1, z_2

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

    def backpropagation(self, x: np.ndarray, a_1: np.ndarray, z_2: np.ndarray,
                        predictions: np.ndarray, targets: np.ndarray):
        """
        Placeholder for the backpropagation process. This should include computing gradients
        and updating parameters using Langevin dynamics.
        """

        # Compute gradients
        grad_weights1, grad_biases1, grad_weights2, grad_biases2 = self.compute_gradients(x, a_1, z_2, predictions,
                                                                                          targets)

        # Langevin dynamics update for weights and biases
        self.weights1 -= self.learning_rate * grad_weights1 + np.sqrt(
            2 * self.learning_rate * self.temperature) * np.random.randn(*self.weights1.shape)
        self.biases1 -= self.learning_rate * grad_biases1 + np.sqrt(
            2 * self.learning_rate * self.temperature) * np.random.randn(*self.biases1.shape)

        self.weights2 -= self.learning_rate * grad_weights2 + np.sqrt(
            2 * self.learning_rate * self.temperature) * np.random.randn(*self.weights2.shape)
        self.biases2 -= self.learning_rate * grad_biases2 + np.sqrt(
            2 * self.learning_rate * self.temperature) * np.random.randn(*self.biases2.shape)

        # Similarly for weights2 and biases2...

    def compute_gradients(self, x: np.ndarray, a_1: np.ndarray, z_2: np.ndarray,
                          predictions: np.ndarray, targets: np.ndarray):
        """
        Placeholder for gradient computation. Actual implementation depends on the network
        architecture and loss function.

        Args:
            x (ndarray): Input data.
            a_1 (ndarray): Activations of the hidden layer.
            z_2 (ndarray): Pre-activation values of the output layer.
            predictions (ndarray): Network output.
            targets (ndarray): True target values.

        Returns:
            A tuple containing gradients for weights and biases in both layers.
        """

        N = targets.shape[0]  # Number of  targets

        # Gradient of loss w.r.t predictions
        d_loss_d_predictions = 2 * (predictions - targets) / N

        # Gradients for output layer (ReLU activation)
        d_predictions_d_z2 = (z_2 > 0).astype(z_2.dtype)
        d_loss_d_z2 = d_loss_d_predictions * d_predictions_d_z2
        d_loss_d_weights2 = np.dot(a_1.T, d_loss_d_z2)
        d_loss_d_biases2 = np.sum(d_loss_d_z2, axis=0)

        # Backpropagation to hidden layer (Quadratic activation)
        d_z2_d_a1 = self.weights2
        d_loss_d_a1 = np.dot(d_loss_d_z2, d_z2_d_a1.T)

        # The derivative computation for quadratic activation
        z_1 = np.dot(x, self.weights1) + self.biases1
        d_a1_d_z1 = 2 * z_1  # Derivative of quadratic activation

        d_loss_d_z1 = d_loss_d_a1 * d_a1_d_z1
        d_loss_d_weights1 = np.dot(x.T, d_loss_d_z1)
        d_loss_d_biases1 = np.sum(d_loss_d_z1, axis=0)

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

    return X_train, X_test, y_train, y_test


def train_model(model, x_train, y_train, epochs):
    """
    Trains the model on the given dataset and plots the training loss.

    Args:
        model (FullyConnectedNetwork): The neural network model to be trained.
        x_train (np.ndarray): Training input data.
        y_train (np.ndarray): Training target labels.
        epochs (int): Number of epochs to train the model.
    """
    # Initialize a list to track the loss over epochs
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0

        for i in range(len(x_train)):
            # Assuming x_train and y_train are numpy arrays and can be indexed into.
            # Adjust the reshaping as needed, especially if working with multidimensional data like images.
            x_sample = x_train[i].reshape(1, -1)
            y_sample = y_train[i].reshape(1, -1)

            predictions, a_1, z_2 = model.forward_pass(x_sample)
            loss = model.compute_loss(predictions, y_sample)
            total_loss += loss

            model.backpropagation(x_sample, a_1, z_2, predictions, y_sample)

        # Compute the average loss for the epoch and append it to the loss_history list
        average_loss = total_loss / len(x_train)
        loss_history.append(average_loss)

        # Optionally print the average loss every few epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}')

    # Plot the loss history after training
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss During Training')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """
    Main function to execute the training and evaluation.
    """
    file = "kindey stone urine analysis.csv"

    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_data(file)

    # Define the network dimensions and hyperparameters
    input_size = X_train.shape[1]  # Number of features
    hidden_size = 4  # Example: 10 neurons in the hidden layer, adjust based on your data
    output_size = 1  # Assuming binary classification
    learning_rate = 0.1  # Example learning rate, adjust based on your training process
    temperature = 1  # Example temperature for Langevin dynamics, adjust as needed

    # Initialize the neural network model
    model = FullyConnectedNetwork(input_size, hidden_size, output_size, learning_rate, temperature)

    # Define the number of epochs for training
    epochs = 100  # Example: 100 epochs, adjust based on your training needs

    # Train the model
    print("Starting training...")
    train_model(model, X_train, y_train, epochs)
    print("Training completed.")

    # Evaluate the model's performance on the test set
    accuracy = model.evaluate_accuracy(X_test, y_test)
    print(f"Model accuracy on test set: {accuracy:.2%}")


if __name__ == "__main__":
    main()
