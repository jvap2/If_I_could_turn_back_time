from modulus.models.meta import ModelMetaData
from modulus.models.module import Module
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.optim import Adam
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from modulus.models.layers.fourier_layers import FourierLayer 
from torch.optim.lr_scheduler import ReduceLROnPlateau

@dataclass
class MetaData(ModelMetaData):
    name: str = "ComplexPINN"
    jit: bool = False
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True

class ComplexPINN(Module):

    def __init__(self):
        super(ComplexPINN, self).__init__(meta=MetaData())
        num_freq_bands = 10  
        gaussian_std = 1.0  
        self.fourier_layer = FourierLayer(1, ["gaussian", gaussian_std, num_freq_bands])
        
        self.real_part = nn.Sequential(
            nn.Linear(self.fourier_layer.out_features(), 128),
            nn.GELU(),
            nn.Linear(128, 64), 
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.imag_part = nn.Sequential(
            nn.Linear(self.fourier_layer.out_features(), 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, inputs):
        fourier_features = self.fourier_layer(inputs)
        fourier_features = torch.tanh(fourier_features)
        real = self.real_part(fourier_features)
        imag = self.imag_part(fourier_features)
        return torch.complex(real, imag)

def analytical_solution_time(t):
    return 5.57 * np.exp(-0.4 * t) * np.sin(1.44 * t)

def complex_loss(output, target):
    return torch.mean(torch.abs(output.real - target.real) + torch.abs(output.imag - target.imag))


def train_model(model, inputs, targets, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = complex_loss(predictions, targets)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")


model = ComplexPINN()
print(model)
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

t = torch.linspace(0, 50, 500).view(-1, 1).float()
y = analytical_solution_time(t.numpy().flatten())
Y = np.fft.fft(y)
Y = torch.view_as_complex(torch.tensor(np.stack((Y.real, Y.imag), axis=-1)))

train_model(model, t, Y)


predictions = model(t).detach().numpy()
freq = np.fft.fftfreq(t.shape[0], d=(t[1] - t[0]).item())

plt.figure(figsize=(10, 6))
plt.plot(freq, np.abs(Y), label='Analytical FT')
plt.plot(freq, np.abs(np.fft.fft(predictions)), label='PINN FT', linestyle='--')
plt.legend()
plt.title('Comparison of FT Magnitude')
plt.xlabel('Frequency ')
plt.ylabel('Magnitude')
plt.grid(True)
plt.savefig('FT_Magnitude.png')
