import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

class ANN:
    def __init__(self, layers, learning_rate=0.1):
        self.lr = learning_rate
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_pred, y_true):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def forward(self, X):
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)

        activations = [X]
        pre_activations = []

        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            a = self.relu(z)
            activations.append(a)

        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        pre_activations.append(z)
        a = self.softmax(z)
        activations.append(a)

        return activations, pre_activations

    def backward(self, X, y_true, activations, pre_activations):
        batch_size = X.shape[0]
        error = activations[-1] - y_true
        gradients_w = []
        gradients_b = []

        for i in range(len(self.weights) - 1, -1, -1):
            grad_w = activations[i].T @ error / batch_size
            grad_b = np.sum(error, axis=0, keepdims=True) / batch_size
            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)

            if i > 0:
                error = (error @ self.weights[i].T) * self.relu_derivative(
                    pre_activations[i - 1]
                )

        return gradients_w, gradients_b

    def update_weights(self, gradients_w, gradients_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * gradients_w[i]
            self.biases[i] -= self.lr * gradients_b[i]

    def train_step(self, X_batch, y_batch):
        activations, pre_activations = self.forward(X_batch)
        loss = self.cross_entropy_loss(activations[-1], y_batch)
        gradients_w, gradients_b = self.backward(
            X_batch, y_batch, activations, pre_activations
        )
        self.update_weights(gradients_w, gradients_b)
        return loss

    def predict(self, X):
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)
        return np.mean(predictions == y)

    def print_confusion_matrix(self, X, y):
        y_pred = self.predict(X)
        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)

        cm = confusion_matrix(y, y_pred)
        
        print("\n--- Confusion Matrix ---")
        print("Rows: True labels | Columns: Predicted labels")
        print(cm)
        
        print("\n--- Classification Report ---")
        print(classification_report(y, y_pred, digits=4))

    def train(self, X_train, y_train, X_val, y_val, epochs=15, batch_size=128):
        num_samples = X_train.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0.0
            num_batches = 0
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]
                epoch_loss += self.train_step(X_batch, y_batch)
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            val_acc = self.accuracy(X_val, y_val)
            print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

def load_data_tf():
    print("Loading MNIST via TensorFlow...")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0

    def to_one_hot(labels):
        one_hot = np.zeros((labels.shape[0], 10))
        one_hot[np.arange(labels.shape[0]), labels] = 1
        return one_hot

    return X_train, to_one_hot(y_train), X_test, to_one_hot(y_test)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data_tf()

    X_val, y_val = X_train[:5000], y_train[:5000]
    X_train, y_train = X_train[5000:], y_train[5000:]

    model = ANN(layers=[784, 128, 32, 10], learning_rate=0.1)

    print("\nStarting Training...\n")
    model.train(X_train, y_train, X_val, y_val, epochs=30, batch_size=128)

    test_acc = model.accuracy(X_test, y_test)
    print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

    model.print_confusion_matrix(X_test, y_test)

    print("\n--- Sample Predictions ---")
    num_samples = 10
    random_indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    for idx in random_indices:
        sample_img = X_test[idx : idx + 1]
        true_label = np.argmax(y_test[idx])
        pred_label = model.predict(sample_img)[0]
        
        status = "Correct" if true_label == pred_label else "Wrong"
        print(f"Sample Index: {idx} | True Label: {true_label} | Predicted: {pred_label} | {status}")