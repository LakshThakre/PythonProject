# TODO : ANN for MNIST Classification
# NOTE : Architecture: 784 -> 128 -> 32 -> 10

import numpy as np
import tensorflow as tf


class ANN:
    def __init__(self, layers, learning_rate=0.1):

        # Initialize network
        # layers: [input_size, hidden1, hidden2, output_size]

        self.lr = learning_rate
        self.weights = []
        self.biases = []
        
        # random weights and zero biases for each layer
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        # ReLU activation: max(0, x)
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        # ReLU derivative: 1 if x > 0, else 0
        return (x > 0).astype(float)
    
    def softmax(self, x):
        # Softmax activation for output layer
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        # Forward propagation
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
    
    def backward(self):
        # Backward propagation.
        pass
    
    def update_weights(self):
        # Update weights using gradients.
        pass
    