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


# Set up RNN for time stepping Runge Kutta Method

class RK_RNN(torch.nn.Module):
    def __init__(self, f, dt, k):
        super().__init__()
        self.f = f
        self.dt = dt
        self.k = k
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.empty(k)) for i in range(k)])
        for i in range(k):
            torch.nn.init.normal_(self.weights[i], std=0.01)
    
    def forward(self, x):
        k = [self.f(x)]
        for i in range(1, self.k):
            x_i = x
            for j in range(i):
                x_i = x_i + self.dt * self.weights[j] * k[j]
            k.append(self.f(x_i))
        x = x + self.dt * sum([self.weights[i] * k[i] for i in range(self.k)])
        return x
    
def bmv(w, x):
    x = x.unsqueeze(-1)
    y = w @ x
    y = y.squeeze(-1)
    return y

## Solve the heat equation using the RK_RNN

rnn = RK_RNN(f, 0.01, 100)

x = torch.linspace(0, 1, 100)
t = torch.linspace(0, 1, 100)
X, T = torch.meshgrid(x, t)

u = rnn(X)

# Plot the solution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, u.detach().numpy(), cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
plt.savefig('heat_eqn.png')