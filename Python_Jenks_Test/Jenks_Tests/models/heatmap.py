import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn

# Define the model architecture (LeNet300_100 in this case)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=784, out_features=300),
    nn.ReLU(),
    nn.Linear(in_features=300, out_features=100),
    nn.ReLU(),
    nn.Linear(in_features=100, out_features=10),
)

# Load the state dictionary (weights) into the model
state_dict = torch.load("2025-03-27_MNIST_LeNet300V100_epoch_300.pth")
model.load_state_dict(state_dict)

# Create a directory to save the heatmaps
os.makedirs("heatmaps", exist_ok=True)

# Iterate through the model's layers and save weights and biases as heatmaps
for i, layer in enumerate(model):
    if isinstance(layer, nn.Linear):  # Only process Linear layers
        # Save weights as a heatmap
        weights = layer.weight.data.cpu().numpy()  # Convert weights to a NumPy array
        plt.figure(figsize=(10, 8))
        plt.imshow(weights, cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.title(f"Layer {i} Weights")
        plt.xlabel("Input Neurons")
        plt.ylabel("Output Neurons")
        plt.savefig(f"heatmaps/layer_{i}_weights.png")
        plt.close()

        # Save biases as a heatmap
        biases = layer.bias.data.cpu().numpy()  # Convert biases to a NumPy array
        plt.figure(figsize=(10, 2))  # Adjust the figure size for biases
        plt.imshow(biases.reshape(1, -1), cmap="viridis", aspect="auto")  # Reshape for visualization
        plt.colorbar()
        plt.title(f"Layer {i} Biases")
        plt.xlabel("Bias Index")
        plt.ylabel("Bias Value")
        plt.savefig(f"heatmaps/layer_{i}_biases.png")
        plt.close()

print("Heatmaps for weights and biases saved in the 'heatmaps' directory.")