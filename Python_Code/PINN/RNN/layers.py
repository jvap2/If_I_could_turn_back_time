import torch


''' 
This code will contain examples of utilizing RNNs in order to solve PDEs.

The first example will be a simple RNN that will be used to solve the heat equation.

The heat eqaution is given by:

    du/dt = alpha * d^2u/dx^2

where alpha is the thermal diffusivity.

The initial condition is given by:

    u(x,0) = f(x)

The boundary conditions are given by:

    u(0,t) = g1(t)
    u(L,t) = g2(t)

The solution to the heat equation is given by:

    u(x,t) = sum(n=1 to infinity) B_n * sin(n*pi*x/L) * exp(-alpha*(n*pi/L)^2*t)

'''

# Define the initial condition
def f(x):
    return torch.sin(x)

# Define the boundary conditions
def g1(t):
    return torch.zeros_like(t)

def g2(t):
    return torch.zeros_like(t)

# Define the solution to the heat equation

def u(x,t,alpha,L):
    u = torch.zeros_like(x)
    for n in range(1,100):
        B_n = 2/(n * torch.pi) * (g1(t) - g2(t))
        u += B_n * torch.sin(n * torch.pi * x / L) * torch.exp(-alpha * (n * torch.pi / L)**2 * t)
    return u


# Set up RNN

class RNN(torch.nn.Module):
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
        """
        if state.size()[-1] != self.dims:
            raise TypeError(f'Previous hidden-state vector must have size {self.dims}')
        if inp.size()[-1] != self.dims:
            raise TypeError(f'Input vector must have size {self.dims}')
        state = torch.tanh(torch.matmul(self.W_hi, inp) + torch.matmul(self.W_hh, state) + self.b)
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
    
# Set up the RNN
rnn = RNN(1)

# Set up the optimizer
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)


# Set up the training data
x = torch.linspace(0,1,100)
t = torch.linspace(0,1,100)
X,T = torch.meshgrid(x,t)
X = X.flatten()
T = T.flatten()
X = X.unsqueeze(1)
T = T.unsqueeze(1)

# Set up the training loop
for epoch in range(1000):
    optimizer.zero_grad()
    u_hat = rnn(torch.cat([X,T],1))
    u_hat = u_hat[:,0]
    u = u(X,T,1,1)
    loss = torch.mean((u_hat - u)**2)
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch}, Loss: {loss.item()}')


# Plot the results
import matplotlib.pyplot as plt
u_hat = rnn(torch.cat([X,T],1))
u_hat = u_hat[:,0]
u_hat = u_hat.reshape(100,100)
plt.imshow(u_hat.detach().numpy())
plt.savefig('heat_eqn_0.png')

u = u(X,T,1,1)
u = u.reshape(100,100)
plt.imshow(u.detach().numpy())
plt.savefig('heat_eqn.png')