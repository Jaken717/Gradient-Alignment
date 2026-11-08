import torch
import torch.nn.functional as F
import math
import random
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
from functorch import make_functional, vmap, jacrev
from torch.func import functional_call, grad, vmap
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import os
from src.arch.FCN_1 import FCN_1
from src.arch.FCN_5 import FCN_5

seed = 888
learning_rate = 2
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device("cuda:0")
width = 5000
batch_size = 1000
Id = 6
train_size = 1000

# @title
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = datasets.MNIST(
    root="./mnist",
    train=True,
    download=True,
    transform=transform
)

# Get indices for each digit
idx_6 = (trainset.targets == 6).nonzero(as_tuple=True)[0]
idx_7 = (trainset.targets == 7).nonzero(as_tuple=True)[0]

# Select exactly 500 samples from each class
idx = torch.cat([
    idx_6[:train_size // 2],
    idx_7[:train_size // 2]
])

# Shuffle to avoid class ordering
idx = idx[torch.randperm(len(idx))]

# Update dataset
trainset.data = trainset.data[idx]
trainset.targets = (trainset.targets[idx] - 6) * 2 - 1

print(f"Size of training set: {len(trainset.targets)}")
print(f"Label +1: {(trainset.targets == 1).float().mean().item():.2f}")
print(f"Label -1: {(trainset.targets == -1).float().mean().item():.2f}")

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle= True)
sample_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle= True)
model = FCN_1(width = width).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epoch = 0
training_loss = []

# for epoch in range(1000):
while True:
    print(f"\n=== Epoch {epoch} ===")
    for batch, (input,label) in enumerate(train_loader):
        # print(f"Batch {batch}")
        optimizer.zero_grad()
        input, label = input.to(device), label.to(device).view(-1,1).float()
        loss = F.mse_loss(model(input), label)
        loss.backward()
        training_loss.append(loss.item())
        optimizer.step()
    epoch += 1
    if training_loss[-1] < 1e-4:   # GD alone never reaches exactly 0
        break

plt.plot(training_loss)
print(f"GD loss: {training_loss[-1]}")

# Finish with Gauss-Newton steps in float64: each step is the min-norm parameter
# change that makes every training output hit its label (quadratic convergence).
model = model.double()
X = torch.stack([trainset[i][0] for i in range(len(trainset))]).to(device).double().view(-1, 784)
Y = trainset.targets.to(device).double()
d, m = 784, width
W, a = model.fc1.weight.data, model.fc5.weight.data.view(-1)
for it in range(10):
    H = X @ W.T / math.sqrt(d)
    Phi, Dm = F.relu(H), (H > 0).double()
    r = Phi @ a / math.sqrt(m) - Y
    print(f"Gauss-Newton iter {it}: loss {(r**2).mean().item():.3e}")
    if (r**2).mean() < 1e-30:
        break
    K = Phi @ Phi.T / m + (X @ X.T / d) * ((Dm * a**2) @ Dm.T) / m   # J J^T
    c = torch.linalg.solve(K, r)
    dW = ((c[:, None] * Dm) * a).T @ X / math.sqrt(m * d)
    a -= Phi.T @ c / math.sqrt(m)
    W -= dW
with torch.no_grad():
    print(f"Final float64 loss: {F.mse_loss(model(X).view(-1), Y).item():.3e}")
# Save the trained model
os.makedirs("./checkpoints", exist_ok=True)
save_path = f"./checkpoints/FCN_1_width{width}_seed{seed}.pt"
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
