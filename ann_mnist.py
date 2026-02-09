"""
Simple ANN for MNIST Classification
Architecture: 784 -> 128 -> 32 -> 10
"""

import numpy as np
import tensorflow as tf


class ANN:
    """Simple Artificial Neural Network"""
    
    def __init__(self, layers, learning_rate=0.1):
        """
        Initialize network
        layers: [input_size, hidden1, hidden2, output_size]
        """
        self.lr = learning_rate
        self.weights = []
        self.biases = []
        
        # Create weights and biases for each layer
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        """ReLU activation: max(0, x)"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """ReLU derivative: 1 if x > 0, else 0"""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax activation for output layer"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """Forward pass through network"""
        # Flatten images: (batch, 28, 28) -> (batch, 784)
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        activations = [X]
        pre_activations = []
        
        # Pass through hidden layers
        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            a = self.relu(z)
            activations.append(a)
        
        # Output layer
        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        pre_activations.append(z)
        a = self.softmax(z)
        activations.append(a)
        
        return activations, pre_activations
    
    def backward(self, X, y_true, activations, pre_activations):
        """Backward pass - compute gradients"""
        batch_size = X.shape[0]
        
        # Start with output layer error
        error = activations[-1] - y_true
        
        gradients_w = []
        gradients_b = []
        
        # Compute gradients for each layer (backwards)
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient for weights and biases
            grad_w = activations[i].T @ error / batch_size
            grad_b = np.sum(error, axis=0, keepdims=True) / batch_size
            
            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)
            
            # Propagate error to previous layer
            if i > 0:
                error = (error @ self.weights[i].T) * self.relu_derivative(pre_activations[i-1])
        
        return gradients_w, gradients_b
    
    def update_weights(self, gradients_w, gradients_b):
        """Update weights using gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * gradients_w[i]
            self.biases[i] -= self.lr * gradients_b[i]
    
    def train_step(self, X_batch, y_batch):
        """Single training step on a batch"""
        activations, pre_activations = self.forward(X_batch)
        gradients_w, gradients_b = self.backward(X_batch, y_batch, activations, pre_activations)
        self.update_weights(gradients_w, gradients_b)
    
    def predict(self, X):
        """Predict class labels"""
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)
    
    def accuracy(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        if len(y.shape) > 1:  # If one-hot encoded
            y = np.argmax(y, axis=1)
        return np.mean(predictions == y)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=15, batch_size=128):
        """Train the network"""
        num_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Train on mini-batches
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                self.train_step(X_batch, y_batch)
            
            # Print progress
            train_acc = self.accuracy(X_train, y_train)
            val_acc = self.accuracy(X_val, y_val)
            print(f"Epoch {epoch+1:2d}/{epochs} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")


def load_data():
    """Load and prepare MNIST data"""
    print("Loading MNIST dataset...")
    
    # Load from TensorFlow
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    # One-hot encode labels
    def to_one_hot(labels):
        one_hot = np.zeros((labels.shape[0], 10))
        one_hot[np.arange(labels.shape[0]), labels] = 1
        return one_hot
    
    y_train = to_one_hot(y_train)
    y_test = to_one_hot(y_test)
    
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # Split off validation set
    X_val = X_train[:5000]
    y_val = y_train[:5000]
    X_train = X_train[5000:]
    y_train = y_train[5000:]
    
    print(f"Training samples:   {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples:       {X_test.shape[0]}")
    
    # Create network: 784 -> 128 -> 32 -> 10
    print("\nCreating network: 784 -> 128 -> 32 -> 10")
    model = ANN(layers=[784, 128, 32, 10], learning_rate=0.1)
    
    # Train
    print("\nTraining...\n")
    model.train(X_train, y_train, X_val, y_val, epochs=15, batch_size=128)
    
    # Test
    test_accuracy = model.accuracy(X_test, y_test)
    print(f"\n{'='*50}")
    print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"{'='*50}")
    
    # Show sample predictions
    print("\nSample Predictions:")
    for i in range(10):
        true_label = np.argmax(y_test[i])
        pred_label = model.predict(X_test[i:i+1])[0]
        symbol = '✓' if true_label == pred_label else '✗'
        print(f"  Sample {i+1}: True = {true_label}, Predicted = {pred_label} {symbol}")