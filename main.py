import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from modulus.models.layers.fourier_layers import FourierLayer, FourierFilter
from torch.optim.lr_scheduler import StepLR


# Analytical solution for the time domain
def analytical_solution_time(t):
    return 5.57 * np.exp(-0.4 * t) * np.sin(1.44 * t)

class FourierFilterPINN(nn.Module):
    def __init__(self, in_features, layer_size, nr_layers, input_scale):
        super(FourierFilterPINN, self).__init__()
        self.fourier_filter = FourierFilter(in_features, layer_size, nr_layers, input_scale)
        
        self.real_part = nn.Sequential(
            nn.Linear(layer_size, 512), 
            nn.GELU(),
            nn.Linear(512, 1)
        )
        self.imag_part = nn.Sequential(
            nn.Linear(layer_size, 512),
            nn.GELU(),
            nn.Linear(512, 1)
        )

    def forward(self, inputs):
        filtered_features = self.fourier_filter(inputs)
        real = self.real_part(filtered_features)
        imag = self.imag_part(filtered_features)
        return torch.complex(real, imag)

def custom_frequency_weighted_loss(predictions, targets, freq, low=-2, high=2, weight=10):
    mask = (freq >= low) & (freq <= high)
    mse = (predictions - targets).abs() ** 2
    weighted_mse = torch.where(mask, weight * mse, mse)
    return torch.mean(weighted_mse)


in_features = 1
layer_size = 512
nr_layers = 4
input_scale = 1.0

# Prepare the data
t_coloc = torch.linspace(0, 100, 5120).view(-1, 1).float()
y = analytical_solution_time(t_coloc.numpy().flatten())
Y = np.fft.fft(y)
scale_factor = np.max(np.abs(Y))
Y_normalized = torch.view_as_complex(torch.tensor(np.stack((np.real(Y), np.imag(Y)), axis=1), dtype=torch.float32)) / scale_factor
t_coloc_normalized = (t_coloc - torch.min(t_coloc)) / (torch.max(t_coloc) - torch.min(t_coloc))


dt = (t_coloc[1] - t_coloc[0]).item()
n = len(t_coloc)
freq = np.fft.fftfreq(n, d=dt)
freq_tensor = torch.tensor(freq, dtype=torch.float32)



# Initialize model and optimizer
model = FourierFilterPINN(in_features, layer_size, nr_layers, input_scale)
optimizer = Adam(model.parameters(),lr=0.001, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

# Training function
def train_model(model, inputs, targets, freq_tensor, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = custom_frequency_weighted_loss(predictions, targets, freq_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

# Train the model
train_model(model, t_coloc_normalized, Y_normalized, freq_tensor, epochs=2000)

# Prediction and plotting
predictions = model(t_coloc_normalized).detach().numpy()
freq = np.fft.fftfreq(t_coloc.shape[0], d=(t_coloc[1] - t_coloc[0]).item())
plt.figure(figsize=(10, 6))
plt.plot(freq, np.abs(Y * scale_factor), label='Analytical FT')
#plt.plot(freq, np.abs(np.fft.fft(predictions) * scale_factor), label='PINN FT', linestyle='--')
plt.legend()
plt.title('Comparison of FT Magnitude')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.savefig('FT_Magnitude.png')
