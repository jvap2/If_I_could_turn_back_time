import torch
import numpy as np
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


## Define convection-diffusion equation as a space time (x,t) dependent PDE
'''
dc/dt + u dc/dx = D d^2c/dx^2, x in [0,1], t in [0,1]
C(x,t=0)= exp(-1/2((x-x_0)/sigma)^2)
C(x=0,t) = 0
C(x=1,t) = 0

The solution to the convection-diffusion equation is given by:
C(x,t) = sigma/(sigma^2 +2Dt)^(1/2) exp(-(x-x_0-Ut)/(2sigma^2+4Dt))

'''
if torch.cuda.is_available():
    print("CUDA is available on your system.")
    device = torch.device("cuda")   
else:
    print("CUDA is not available on your system.")
    device = torch.device("cpu")

x_0 = .5
D = 0.1
U = 1
sigma = 0.1
x_dim = 512
x = np.linspace(0,1,512)
time_steps = 100
t = np.linspace(0,1,time_steps)

def true_sol(x,t,sigma,D,U,x_0):
    arg = -(x - x_0- U*t)**2/(2*sigma**2 + 4*D*t)
    return sigma/(sigma**2 + 2*D*t)**(1/2) * torch.exp(arg)

def true_sol_cpu(x,t,sigma,D,U,x_0):
    arg = -(x - x_0- U*t)**2/(2*sigma**2 + 4*D*t)
    return sigma/(sigma**2 + 2*D*t)**(1/2) * np.exp(arg)

def initial_condition(x,x_0,sigma):
    return np.exp(-1/2*((x-x_0)/sigma)**2)

def boundary_condition(t):
    return np.zeros_like(t)

def boundary_conditions_gpu(t):
    return torch.zeros_like(t).to(device)

class ConvectionDiffusionPDE(nn.Module):
    def __init__(self, D, U, sigma,x_0,x_dim):
        super(ConvectionDiffusionPDE, self).__init__()
        self.D = D
        self.U = U
        self.sigma = sigma
        self.x_0 = x_0
        self.linear1 = nn.Linear(x_dim, 128)  # First linear layer
        self.linear2 = nn.Linear(128,x_dim)  # Second linear layer
    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        return x
    def loss(self, x, t, model_result,w_1,w_b1,w_b2):
        return w_1*mse_loss(model_result, true_sol(x,t,self.sigma,self.D,self.U,0))+w_b1*mse_loss(model_result[0],boundary_conditions_gpu(t))+w_b2*mse_loss(model_result[-1],boundary_conditions_gpu(t))
    

# Set up the model

model = [ConvectionDiffusionPDE(D,U,sigma,x_0,x_dim).to(device) for _ in range(time_steps)]


epochs = 1000

# Set up the data
x = torch.tensor(x).float()
t = torch.tensor(t).float()
x_sol, t_sol = np.meshgrid(x, t)
# x = x.reshape(-1)
# t = t.reshape(-1)
x_=x.to(device)
t_=t.to(device)
# true = torch.tensor(true_sol(x_sol, t_sol, 0.1, 0.1, 1, 0.5)).float().to(device)
# print(type(true))
output = []
print(t.shape[0])
w_1 = .5
w_b1 = .25
w_b2 = .25
for i in range(t.shape[0]):
    optimizer = Adam(model[i].parameters(), lr=0.0001)
    for j in range(epochs):
        res = model[i](x_)
        optimizer.zero_grad()
        loss = model[i].loss(x_,t_[i],res,w_1,w_b1,w_b2)
        loss.backward()
        optimizer.step()
        print(f'Epoch {j}, Loss {loss.item()}')
        if j==epochs-1:
            output.append(res)
output = torch.stack(output)
print(output.shape)

true = true_sol_cpu(x_sol,t_sol,sigma,D,U,x_0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X, T = np.meshgrid(x_sol, t_sol)
Z_true = true
Z_pred = output.cpu().detach().numpy()

ax.plot_surface(x_sol, t_sol, Z_true, label='True')
ax.plot_surface(x_sol, t_sol, Z_pred, label='Predicted')

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('C')

plt.legend()
plt.savefig('convection_diffusion.png')

