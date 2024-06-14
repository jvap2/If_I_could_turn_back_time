import torch
import numpy as np
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import pandas as pd
import sys
import math
import scipy
from scipy.integrate import quad


# x_steps = int(sys.argv[1])
# t_steps = int(sys.argv[2])
# hidden_size = int(sys.argv[3])
x_steps = 512
t_steps = 10
hidden_size = 256
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

x_0 = 0
x_L = -4
x_R = 4
t_0 = 0
t_L = 1
D = 1
U = 5
x = np.linspace(x_L,x_R,x_steps)
t = np.linspace(t_0,t_L,t_steps)
K = .01
Beta = 0.2

# def true_sol(x,t,Beta,K):
#     '''
#     Input:
#     x: spatial coordinate
#     t: temporal coordinate
#     Beta: floating point value to vary order of fractional derivative
#     K: function of x and t, which is a constant for this solution
#     '''
#     # arg = lambda eps: torch.exp(-K*t*torch.cos(Beta*np.pi/2))*(eps**(2-Beta))*torch.cos(x*eps)
#     # arg = lambda eps: torch.exp(-K*t*torch.cos(torch.tensor(Beta*np.pi/2)))*(eps**(2-Beta))*torch.cos(x*eps)
#     print(type(x))
#     u_true= lambda ep,x,t: np.exp(-K*t*np.cos(math.pi/10)*(ep**(2-Beta)))*np.cos(x*ep)
#     int=(1/math.pi)*(quad(u_true,0,np.inf,args=(x,t))[0])
#     sol = torch.zeros(x.shape[0],t.shape[0])
#     for i, x_i in x:
#         for j, t_j in t:
#             sol[i,j] = int(x_i,t_j)
#     return sol.to(device)


def true_sol(x,t,Beta,K):
    u_true= lambda ep,x,t: np.exp(-K*t*np.cos(math.pi/10)*(ep**(2-Beta)))*np.cos(x*ep)
    sol = torch.zeros(x.shape[0],)
    for i, x_i in enumerate(x):
        int_val = (1/math.pi)*(quad(u_true,0,10e3,args=(x_i.item(),t.item()))[0])+\
        (1/math.pi)*(quad(u_true,10e3,10e6,args=(x_i.item(),t.item()))[0])
        sol[i] = torch.tensor(int_val)
    return sol.to(device)

def true_sol_cpu(x,t,Beta,K):
    u_true= lambda ep,x,t: np.exp(-K*t*np.cos(math.pi/10)*(ep**(2-Beta)))*np.cos(x*ep)
    sol = np.zeros((x.shape[0],t.shape[0]))
    for i, x_i in enumerate(x):
        for j, t_j in enumerate(t):
            int_val = (1/math.pi)*(quad(u_true,0,10e3,args=(x_i,t_j))[0]) + \
                       (1/math.pi)*(quad(u_true,10e3,10e6,args=(x_i,t_j))[0])
            sol[i,j] = int_val
    return sol


def initial_condition(x,x_0):
    ''' Initial conditions are the dirac delta function centered at x_0'''
    return np.where(x==x_0,1,0)

def initial_condition_gpu(x,x_0):
    return torch.where(x==x_0,1,0).to(device)

def boundary_condition(t):
    return np.zeros_like(t)

def boundary_conditions_gpu(t):
    return torch.zeros_like(t).to(device)

class ConvectionDiffusionPDE(nn.Module):
    def __init__(self, D, U,x_0,x_dim,hidden_size):
        super(ConvectionDiffusionPDE, self).__init__()
        self.K = D
        self.Beta = U
        self.x_0 = x_0
        self.linear1 = nn.Linear(x_dim, hidden_size)  # First linear layer
        self.linear2 = nn.Linear(hidden_size,x_dim)  # Second linear layer
    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        return x
    def loss(self, x, t, model_result,w_1,w_b1,w_b2, x_0):
        return w_1*mse_loss(model_result, true_sol(x,t,self.Beta,self.K))+w_b1*mse_loss(model_result[0],boundary_conditions_gpu(t))+w_b2*mse_loss(model_result[-1],boundary_conditions_gpu(t))
    

# Set up the model

model = [ConvectionDiffusionPDE(K,Beta,x_0,x_steps,hidden_size).to(device) for _ in range(t_steps)]


epochs = 2

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
count = 0
time_overall = 0
for i in range(t.shape[0]):
    optimizer = Adam(model[i].parameters(), lr=0.0001)
    for j in range(epochs):
        count+=1
        start = time.time()
        res = model[i](x_)
        stop = time.time()
        time_overall += stop-start
        optimizer.zero_grad()
        loss = model[i].loss(x,t[i],res,w_1,w_b1,w_b2,x_0)
        loss.backward()
        optimizer.step()
        print(f'Epoch {j}, Loss {loss.item()}')
        if j==epochs-1:
            output.append(res)
output = torch.stack(output)
print(output.shape)
ave_time = time_overall/count
true = true_sol_cpu(x,t,Beta,K)


# l2_error = np.mean(np.power(output.cpu().detach().numpy() - true, 2))
# l2_error = np.sqrt(l2_error)
# print(f"L2 Error: {l2_error.item()}")
# weight_matrices = []
# for i in range(t_steps):
#     weight_matrices.append(model[i].state_dict())

# ## Extract weight matrices and find their condition number
# cond_first_layer = []
# cond_second_layer = []
# for i in range(t_steps):
#     for key in weight_matrices[i]:
#         if key == 'linear1.weight':
#             cond_first_layer.append(np.linalg.cond(weight_matrices[i][key].cpu().numpy()))
#         if key == 'linear2.weight':
#             cond_second_layer.append(np.linalg.cond(weight_matrices[i][key].cpu().numpy().T))
# ## Find the mean and standard deviation of the condition numbers
# mean_cond_first_layer = np.mean(cond_first_layer)
# std_cond_first_layer = np.std(cond_first_layer)
# mean_cond_second_layer = np.mean(cond_second_layer)
# std_cond_second_layer = np.std(cond_second_layer)

# file = 'convection_diffusion.csv'

# df = pd.read_csv(file)
# trial = df.shape[0]
# df.loc[trial-1,'L2Error'] = l2_error.item()
# df.loc[trial-1,'AverageTimeperEpoch'] = ave_time
# df.loc[trial-1,'MeanConditionNumberFirstLayer'] = mean_cond_first_layer
# df.loc[trial-1,'StdConditionNumberFirstLayer'] = std_cond_first_layer
# df.loc[trial-1,'MeanConditionNumberSecondLayer'] = mean_cond_second_layer
# df.loc[trial-1,'StdConditionNumberSecondLayer'] = std_cond_second_layer
# df.loc[trial-1,'XSteps'] = x_steps
# df.loc[trial-1,'TSteps'] = t_steps
# df.loc[trial-1,'HiddenSize'] = hidden_size
# df.loc[trial-1,'Epochs'] = epochs
# df.loc[trial-1,'D'] = D
# df.loc[trial-1,'U'] = U
# df.loc[trial-1,'Sigma'] = sigma
# df.loc[trial-1,'X_0'] = x_0
# df.to_csv(file,index=False)




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X, T = np.meshgrid(x_sol, t_sol)
Z_true = true
Z_pred = output.cpu().detach().numpy()

# ax.plot_surface(x_sol, t_sol, Z_true, label='True')
# ax.plot_surface(x_sol, t_sol, Z_pred, label='Predicted')
# ax.plot_surface(x_sol, t_sol, Z_true.T, label='True')



# plt.legend()

from matplotlib import cm
import matplotlib.patches as mpatches

# Your plot_surface calls without the label argument
surf1 = ax.plot_surface(x_sol, t_sol, Z_pred, cmap=cm.coolwarm)
surf2 = ax.plot_surface(x_sol, t_sol, Z_true.T, cmap=cm.inferno)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('C')
# Create a legend using proxy artists
predicted_patch = mpatches.Patch(color=surf1.get_facecolor()[0], label='Predicted')
true_patch = mpatches.Patch(color=surf2.get_facecolor()[0], label='True')
plt.legend(handles=[predicted_patch, true_patch])

plt.savefig('convection_diffusion.png')

