import torch
import torch.nn as nn
import torch.nn.functional as F



class RKIntegrator(torch.nn.Module):
    def __init__(self, dt, method='rk4'):
        super(RKIntegrator, self).__init__()
        self.dt = dt
        if method == 'rk4':
            self.weights = torch.tensor([1/6, 1/3, 1/3, 1/6])
            self.k = 4
        elif method == 'rk2':
            self.weights = torch.tensor([1/2, 1/2])
            self.k = 2
        else:
            raise ValueError('Invalid method')

    def forward(self, f, x):
        k = [f(x)]
        for i in range(1, self.k):
            x_i = x
            for j in range(i):
                x_i = x_i + self.dt * self.weights[j] * k[j]
            k.append(f(x_i))
        x = x + self.dt * sum([self.weights[i] * k[i] for i in range(self.k)])
        return x
    
class RNN(torch.nn.Module):
    """Simple recurrent neural network.

    The constructor takes one argument:
        dims: Size of both the input and output vectors (int)

    The resulting RNN object can be used in two ways:
      - On a whole sequence at once, by calling the object (see documentation for forward())
      - Step by step, using start() and step(); please see the documentation for those methods.

    This implementation adds a _residual connection_, which just means
    that output vector is the standard output vector plus the input
    vector. This helps against overfitting, but makes the
    implementation slightly more complicated.
    """

    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.h0 = torch.nn.Parameter(torch.empty(dims))
        self.W_hi = torch.nn.Parameter(torch.empty(dims, dims))
        self.W_hh = torch.nn.Parameter(torch.empty(dims, dims))
        self.b = torch.nn.Parameter(torch.empty(dims))
        torch.nn.init.normal_(self.h0, std=0.01)
        torch.nn.init.normal_(self.W_hi, std=0.01)
        torch.nn.init.normal_(self.W_hh, std=0.01)
        torch.nn.init.normal_(self.b, std=0.01)

    def start(self):
        """Return the initial state."""
        return self.h0

    def step(self, state, inp):
        """Given the old state, read in an input vector (inp) and
        compute the new state and output vector (out).

        Arguments:
            state:  State (Tensor of size dims)
            inp:    Input vector (Tensor of size dims)

        Returns: (state, out), where
            state:  State (Tensor of size dims)
            out:    Output vector (Tensor of size dims)
        """

        if state.size()[-1] != self.dims:
            raise TypeError(f'Previous hidden-state vector must have size {self.dims}')
        if inp.size()[-1] != self.dims:
            raise TypeError(f'Input vector must have size {self.dims}')

        state = torch.tanh(bmv(self.W_hi, inp) + bmv(self.W_hh, state) + self.b)
        return (state, state + inp)

    def forward(self, inputs):
        """Run the RNN on an input sequence.
        Argument:
            Input vectors (Tensor of size n,dims)

        Return:
            Output vectors (Tensor of size n,dims)
        """

        if inputs.ndim != 2:
            raise TypeError("inputs must have exactly two axes")
        if inputs.size()[1] != self.dims:
            raise TypeError(f'Input vectors must have size {self.dims}')

        h = self.start()
        outputs = []
        for inp in inputs:
            h, o = self.step(h, inp)
            outputs.append(o)
        return torch.stack(outputs)