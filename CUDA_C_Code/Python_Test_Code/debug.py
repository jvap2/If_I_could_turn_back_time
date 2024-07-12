import numpy as np

import torch

from torch import nn


W_Layer_1 = np.array([[-0.301304,-0.0968752,0.369422,0.285944,-0.126029,0.0469441,0.0710443,-0.0424175,0.167604,-0.358093],

[0.0275555,0.230131,-0.175044,0.440262,0.068845,0.337748,0.221647,0.115301,-0.415532,0.221642],

[0.298058,0.380468,0.333864,0.296089,0.428819,0.218071,0.360782,0.432541,0.149262,-0.00245205],

[-0.300556,0.295172,0.347886,-0.378347,0.133902,-0.225356,0.11581,-0.242267,0.17944,-0.163799],

[-0.153146,-0.240218,-0.380882,0.119023,-0.247169,0.135177,0.00955784,0.421692,-0.196736,0.0412392],

[0.19612,-0.345892,-0.025506,0.0827702,0.39741,-0.0439007,-0.146372,0.310978,-0.0585731,-0.444323],

[-0.138687,0.0880843,0.298062,-0.238014,0.15695,-0.0152497,-0.0161571,-0.174453,0.189697,-0.283931],

[0.108962,-0.410663,-0.0769353,0.175293,0.155574,0.123109,-0.136744,-0.282082,0.0975873,0.113733],

[0.206371,-0.153507,0.215055,-0.266349,0.376477,0.165251,0.136964,-0.217108,0.0290161,-0.368823],

[-0.214218,0.337542,0.166475,-0.36337,-0.347686,-0.123788,0.068594,0.0833709,0.148973,-0.188923],

[0.246654,-0.189279,-0.152373,-0.277495,0.433228,-0.444013,0.292828,-0.15073,-0.278881,-0.0567988],

[0.410217,0.374703,0.236908,0.178058,-0.33886,0.166172,-0.103904,0.245318,0.396277,0.372326],

[0.323709,-0.265155,0.262655,0.0429701,-0.181311,0.362183,0.366396,0.334497,-0.00166014,0.0681549],

[-0.30164,-0.20222,0.326089,-0.00679868,-0.0325015,0.312103,-0.00359792,-0.186888,-0.28584,0.164734],

[0.203527,-0.322836,0.0922233,-0.00677821,0.302436,0.200577,-0.28782,-0.248682,-0.00131887,-0.338756],

[-0.323569,-0.124824,-0.156697,0.386299,0.36536,0.109205,0.301268,0.284542,-0.00351116,-0.147606]])

b_Layer_1 = np.array([-0.298888, 0.449242, 0.307967, -0.681916, -0.986471, -1.20903, -1.10917, 0.416365, -0.385802, -0.598864, -0.476913, -1.15641, -0.205548, 1.22894, 0.236372, -0.663376])

W_Layer_2 = np.array([[0.111158,0.35023,0.308194,-0.124068,0.264677,0.0630433,0.0974188,0.183369,0.194752,0.208533,-0.167737,0.0738069,-0.0208142,-0.235499,0.208943,0.258154],

[0.263766,0.116259,-0.0618842,0.0791823,0.068518,0.102956,0.0272636,-0.24866,0.0558766,-0.330245,0.142065,0.0128345,0.23519,0.0106413,-0.273899,-0.00720543],

[0.00731787,-0.319259,0.22228,-0.0815588,0.0973378,-0.0338546,-0.251743,-0.0614633,-0.178875,-0.065926,-0.34121,0.153864,0.0521287,0.221287,0.0584652,-0.0376587],

[-0.0160079,0.350134,-0.31203,-0.301043,0.0995368,0.0687872,-0.19615,-0.19814,0.0920958,0.299469,0.168248,-0.0262674,-0.0434434,0.247902,0.320081,0.317428],

[0.282196,0.188807,-0.117684,0.0259809,-0.198601,-0.0158737,0.318071,-0.0239224,0.271754,0.330415,-0.223612,-0.0296709,0.198148,0.188407,0.286224,-0.171413],

[0.184988,0.327747,-0.118903,-0.0690286,0.0429811,0.0385007,0.0863849,-0.218476,-0.015584,-0.0989206,0.10881,0.294526,-0.204572,0.0753367,0.258401,-0.275929],

[-0.0894095,-0.212837,0.103605,0.0655431,0.124842,0.068123,-0.311933,0.0430428,0.0449843,-0.181991,-0.340181,-0.110421,-0.347138,0.299596,0.0717195,0.191404],

[0.27379,0.30637,-0.231178,-0.0367826,-0.00868279,0.20876,0.0982943,0.329287,-0.243714,-0.14645,0.270259,-0.0947326,0.282441,0.175106,-0.017108,-0.160522]])

b_Layer_2 = np.array([1.26329, -1.06822, 1.0343, 0.348446, 0.618482, 1.20078, -0.893596, -0.615795])

W_Layer_3 = np.array([[-0.398546,-0.342725,-0.255851,-0.363829,0.0891186,-0.441948,0.389553,0.445502],

[-0.443978,0.42522,-0.03095,-0.243031,0.0870112,-0.331163,0.0845852,-0.0236453],

[0.315549,0.426068,0.0265226,0.0822501,0.229398,-0.274764,-0.235828,0.133585],

[0.0381753,-0.483349,0.431518,-0.152454,-0.294286,0.0226287,-0.0990146,-0.192832]])

b_Layer_3 = np.array([0.508845, 0.410501, -0.160262, -0.653303])

inpt = np.array([0.0597015, 0.303371, 0.0927835, 0.669725, 0.333333, 0.102458, 1, 0.333333, 0.525, 0  ], dtype=np.float32)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(10, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 4)
        self.layer1.weight = nn.Parameter(torch.tensor(W_Layer_1, dtype=torch.float32))
        self.layer1.bias = nn.Parameter(torch.tensor(b_Layer_1, dtype=torch.float32))
        self.layer2.weight = nn.Parameter(torch.tensor(W_Layer_2, dtype=torch.float32))
        self.layer2.bias = nn.Parameter(torch.tensor(b_Layer_2, dtype=torch.float32))
        self.layer3.weight = nn.Parameter(torch.tensor(W_Layer_3, dtype=torch.float32))
        self.layer3.bias = nn.Parameter(torch.tensor(b_Layer_3, dtype=torch.float32))
        self.layer1.requires_grad = True
        self.layer2.requires_grad = True
        self.layer3.requires_grad = True
        self.sm = nn.Softmax(dim=0)
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        x = self.layer1(x)
        print(x)
        x= self.relu_1(x)
        print(x)
        x = self.layer2(x)
        print(x)
        x= self.relu_2(x)
        print(x)
        x = self.layer3(x)
        print(x)
        x = self.sm(x)
        print(x)
        return x
    
def hook_fn(module, grad_input, grad_output):
    print("Gradient output:", grad_output)
    

model = Network()


model.sm.register_backward_hook(hook_fn)
model.relu_1.register_backward_hook(hook_fn)
model.relu_2.register_backward_hook(hook_fn)
model.train()
output=model(inpt)
print(output)

## Use categorical cross entropy loss

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=.001)
target = torch.tensor([1.0,0.0,0.0,0.0])

## Perform backward pass

output = output.unsqueeze(0)
target = target.unsqueeze(0)
optimizer.zero_grad()
loss_val = loss(output, target)
loss_val.backward()
optimizer.step()

print(loss_val)



# # # Compute the gradients

## See the updated weights

print(model.layer1.weight)
print(model.layer1.bias)
print(model.layer2.weight)
print(model.layer2.bias)
print(model.layer3.weight)
print(model.layer3.bias)



print("Layer 1 Weights Gradient")
print(model.layer1.weight.grad)
print("Layer 1 bias Gradient")
print(model.layer1.bias.grad)
print("Layer 2 Weights Gradient")
print(model.layer2.weight.grad)
print("Layer 2 bias Gradient")
print(model.layer2.bias.grad)
print("Layer 3 Weights Gradient")
print(model.layer3.weight.grad)
print("Layer 3 bias Gradient")
print(model.layer3.bias.grad)


