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
from torch.nn.utils import parameters_to_vector

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
model_train = FCN_1(width = width).to(device)
optimizer = torch.optim.SGD(model_train.parameters(), lr=learning_rate)
fine_model = FCN_1(width = width).to(device)
state = torch.load(f"./checkpoints/FCN_1_width{width}_seed{seed}.pt", map_location=device)
fine_model.load_state_dict(state)
fine_model.eval()


def sample_gradients(model, loader):
    sample_vectors = []
    for batch, (input,label) in enumerate(loader):
        optimizer.zero_grad()
        gradient_vector = []
        input, label = input.to(device), label.to(device).view(-1,1).float()
        loss = F.mse_loss(model(input), label)
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                gradient_vector.append(param.grad.view(-1))  # Flatten and collect gradient
            else:
                # If there's no gradient, add a zero tensor with the same shape to maintain consistency
                    gradient_vector.append(torch.zeros_like(param).view(-1))
        data_vector = torch.cat(gradient_vector)
        sample_vectors.append(data_vector)
    return sample_vectors

def check_alignment(sample_vectors, directions):
    align = 0
    for v in sample_vectors:
        cos = F.cosine_similarity(-v, directions, dim=0)
        if cos >= 0:
            align += 1
    # print(f"sample number: {len(sample_vectors)}")
    return align/len(sample_vectors)

epoch = 0
training_loss = []
# aligns = 0

for epoch in range(1000):
# while True:
    print(f"\n=== Epoch {epoch} ===")
    for batch, (input,label) in enumerate(train_loader):
        # print(f"Batch {batch}")
        sample_vectors = sample_gradients(model_train, sample_loader)
        with torch.no_grad():
            theta_train = parameters_to_vector(model_train.parameters())
            theta_fine  = parameters_to_vector(fine_model.parameters())
            direction = theta_fine - theta_train   # vector pointing from model_train to fine_model
        align = check_alignment(sample_vectors, direction)
        print(f"align_rate is {align}")
        optimizer.zero_grad()
        input, label = input.to(device), label.to(device).view(-1,1).float()
        loss = F.mse_loss(model_train(input), label)
        loss.backward()
        training_loss.append(loss.item())
        optimizer.step()
    epoch += 1
    if training_loss[-1] < 0.0001:
        break
        
# plt.plot(training_loss)
# print(training_loss[-1])
# Save the trained model
# print(f"alignment number:{aligns}")