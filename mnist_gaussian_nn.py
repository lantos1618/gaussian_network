import time
import psutil
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple

#######################################
# Part 1: Train a Simple MNIST Model  #
#######################################

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 2  # shorter training for demo
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {100 * correct / total:.2f}%')

#######################################
# Part 2: Gaussian Representation of Weights
#######################################

def gaussian_2d(
    x: np.ndarray,
    y: np.ndarray,
    px: float,
    py: float,
    sigma: float,
    amplitude: float
) -> np.ndarray:
    dx = x - px
    dy = y - py
    dist_sq = dx*dx + dy*dy
    return amplitude * np.exp(-dist_sq / (2 * sigma * sigma))

# Extract weights from fc1
fc1_weights = model.fc1.weight.data.cpu().numpy()  # shape [128, 784]
# Flatten them
weights_flat = fc1_weights.flatten()  # ~100k weights

# For demonstration, let's take a subset (e.g., first 20,000) to keep computations manageable
N = min(20000, weights_flat.size)
weights_subset = weights_flat[:N]

# Map these weights to a 2D grid
side = int(np.ceil(np.sqrt(N)))
indices = np.arange(N)
px = (indices % side).astype(np.float32) / side
py = (indices // side).astype(np.float32) / side

sigma = 0.01
res = 256
x_lin = np.linspace(0, 1, res, dtype=np.float32)
y_lin = np.linspace(0, 1, res, dtype=np.float32)
X, Y = np.meshgrid(x_lin, y_lin, indexing='ij')

process = psutil.Process(os.getpid())
mem_start = process.memory_info().rss / (1024 * 1024)
print(f"Memory before Gaussian field computation: {mem_start:.2f} MB")

start = time.time()
field = np.zeros((res, res), dtype=np.float32)

for i in range(N):
    field += gaussian_2d(X, Y, float(px[i]), float(py[i]), sigma, float(weights_subset[i]))

end = time.time()
mem_end = process.memory_info().rss / (1024 * 1024)
print(f"Computation Time: {end - start:.4f}s")
print(f"Memory after Gaussian field computation: {mem_end:.2f} MB")

center_x = res // 2
center_y = res // 2
print("Field sample (center 5x5 block):")
print(field[center_x:center_x+5, center_y:center_y+5])
