import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(0)
torch.manual_seed(0)

def generate_data(num_samples):
    t = np.linspace(0, 10, 1000, endpoint=False)
    signals = [5.57 * np.exp(-0.4 * t) * np.sin(1.44 * t + np.random.randn()) for _ in range(num_samples)]
    ffts = [fft(signal) for signal in signals]
    return np.array(signals), np.array(ffts)

num_samples = 1000
signals, ffts = generate_data(num_samples)
signals = np.real(signals)  # Ensure signals are real
ffts_real = np.real(ffts)  # Real part of the full spectrum
ffts_imag = np.imag(ffts)  # Imaginary part of the full spectrum

# combine real and imaginary parts vertically
ffts_combined = np.hstack((ffts_real, ffts_imag))

# split data into training and testing
split_index = int(0.8 * num_samples)
train_signals, test_signals = signals[:split_index], signals[split_index:]
train_ffts, test_ffts = ffts_combined[:split_index], ffts_combined[split_index:]

# Convert numpy arrays to torch tensors
train_signals = torch.tensor(train_signals, dtype=torch.float32)
test_signals = torch.tensor(test_signals, dtype=torch.float32)
train_ffts = torch.tensor(train_ffts, dtype=torch.float32)
test_ffts = torch.tensor(test_ffts, dtype=torch.float32)

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(2000, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2000)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ANN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_ffts)
    loss = criterion(outputs, train_ffts)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    predictions = model(train_ffts)

def plot_fft_comparison(index):
    actual_fft = test_ffts[index][:1000] + 1j * test_ffts[index][1000:]  # Reconstruct complex FFT from halves
    predicted_fft = predictions[index][:1000] + 1j * predictions[index][1000:]  # Reconstruct complex FFT
    frequencies = fftfreq(1000, 0.01)  # Full range of frequencies including negative

    plt.figure(figsize=(14, 7))
    plt.plot(np.abs(actual_fft), label='Actual FFT', linewidth=2)
    plt.plot(np.abs(predicted_fft), label='Predicted FFT', linestyle='--')
    plt.title('Comparison of Actual and Predicted FFT Magnitudes')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.xlim(-50, 50)  # limits to -50 to 50 Hz
    plt.show()

plot_fft_comparison(10)  # index as needed for different samples
plt.savefig('FT_Magnitude_ANN.png')
