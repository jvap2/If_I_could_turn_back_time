import numpy as np


class ReLU:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)

    def backward(self, dout):
        dx = dout * (self.cache > 0)
        return dx

class Sigmoid:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        self.cache = 1 / (1 + np.exp(-x))
        return self.cache

    def backward(self, dout):
        sigmoid = self.cache
        dx = dout * sigmoid * (1 - sigmoid)
        return dx



class Linear:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.01  # Initialize weights
        self.b = np.zeros((1, output_dim))  # Initialize biases
        self.cache = None  # To store the input x for backward pass

    def forward(self, x):
        self.cache = x  # Cache the input x
        out = np.dot(x, self.W) + self.b  # Linear transformation
        return out

    def backward(self, dout):
        x = self.cache  # Retrieve the cached input

        # Compute gradient with respect to input
        dx = np.dot(dout, self.W.T)  # Gradient of the loss w.r.t. input x

        # Compute gradient with respect to weights
        dW = np.dot(x.T, dout)  # Gradient of the loss w.r.t. weights W

        # Compute gradient with respect to biases
        db = np.sum(dout, axis=0, keepdims=True)  # Gradient of the loss w.r.t. biases b

        # Update weights and biases (if using a simple gradient descent step)
        learning_rate = 0.01  # Example learning rate
        self.W -= learning_rate * dW  # Update weights
        self.b -= learning_rate * db  # Update biases

        return dx  # Return the gradient with respect to input for the next layer in backpropagation
    
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

