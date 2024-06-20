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
        dx = np.dot(dout, self.W.T)  # Gradient of the loss w.r.t. input x, delta* W^T

        # Compute gradient with respect to weights
        dW = np.dot(x.T, dout)  # Gradient of the loss w.r.t. weights W, delta* x^T(activation of previous layer)

        # Compute gradient with respect to biases
        db = np.sum(dout, axis=0, keepdims=True)  # Gradient of the loss w.r.t. biases b

        # Update weights and biases (if using a simple gradient descent step)
        learning_rate = 0.01  # Example learning rate
        self.W -= learning_rate * dW  # Update weights
        self.b -= learning_rate * db  # Update biases

        return dx  # Return the gradient with respect to input for the next layer in backpropagation

class Conv2D:
    def __init__(self, num_filters, filter_size, input_channels):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels
        self.W = np.random.randn(filter_size, filter_size, input_channels, num_filters) * 0.01  # Initialize weights
        self.b = np.zeros((1, 1, 1, num_filters))  # Initialize biases
        self.cache = None  # To store the input and other necessary variables for backward pass

    def forward(self, x):
        self.cache = x  # Cache the input x
        # Add the forward pass implementation here (not shown for brevity)
        # Example: out = convolution_operation(x, self.W, self.b)
        # self.cache = (x, self.W, self.b)  # Cache necessary values
        # return out
        filter_size, num_filters = self.filter_size, self.num_filters

        # Retrieve dimensions
        n_H_prev, n_W_prev, n_C_prev = x.shape[1], x.shape[2], x.shape[3]
        n_H = n_H_prev - filter_size + 1
        n_W = n_W_prev - filter_size + 1

        # Initialize output
        out = np.zeros((x.shape[0], n_H, n_W, num_filters))

        # Compute the convolution

        for i in range(n_H):
            for j in range(n_W):
                for c in range(num_filters):
                    h_start = i
                    h_end = h_start + filter_size
                    w_start = j
                    w_end = w_start + filter_size

                    out[:, i, j, c] = np.sum(x[:, h_start:h_end, w_start:w_end, :] * self.W[:, :, :, c] + self.b[:, :, :, c], axis=(1, 2, 3))
        
        return out  # Return the output

    def backward(self, dout):
        x = self.cache  # Retrieve the cached input
        filter_size, num_filters = self.filter_size, self.num_filters

        # Initialize gradients
        dx = np.zeros_like(x)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        # Retrieve dimensions
        n_H_prev, n_W_prev, n_C_prev = x.shape[1], x.shape[2], x.shape[3]
        n_H, n_W, n_C = dout.shape[1], dout.shape[2], dout.shape[3]

        # Compute gradients
        for i in range(n_H):
            for j in range(n_W):
                for c in range(n_C):
                    # Compute the gradient with respect to the input (dx)
                    h_start = i
                    h_end = h_start + filter_size
                    w_start = j
                    w_end = w_start + filter_size

                    dx[:, h_start:h_end, w_start:w_end, :] += self.W[:, :, :, c] * dout[:, i, j, c][:, None, None, None]

                    # Compute the gradient with respect to the weights (dW)
                    dW[:, :, :, c] += np.sum(x[:, h_start:h_end, w_start:w_end, :] * dout[:, i, j, c][:, None, None, None], axis=0)

                    # Compute the gradient with respect to the biases (db)
                    db[:, :, :, c] += np.sum(dout[:, i, j, c], axis=0)

        return dx, dW, db  # Return the gradients



    
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



# Define the network architecture
layers = [
    Linear(input_dim=4, output_dim=5),
    ReLU(),
    Linear(input_dim=5, output_dim=3),
    Sigmoid()
]

# Create the neural network
nn = NeuralNetwork(layers)

# Forward pass
x = np.random.randn(10, 4)  # Example input
output = nn.forward(x)

# # Compute loss (example, assuming some loss function)
# loss = compute_loss(output, y)

# # Backward pass
# dout = compute_loss_gradient(output, y)  # Gradient of loss w.r.t. output
# nn.backward(dout)
