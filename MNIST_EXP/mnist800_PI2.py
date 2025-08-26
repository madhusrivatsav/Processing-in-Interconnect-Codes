import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import torch.optim.lr_scheduler as lr_scheduler
import math

class MPLayer_in_K(torch.nn.Module):
  def __init__(self,inp_node,out_node,gamma,diff=0):
    super().__init__()
    self.inp_node = inp_node
    self.out_node = out_node
    self.gamma = gamma
    self.diff = diff # differential inputs are given or not
    torch.manual_seed(43)
    self.weight = torch.nn.Parameter(torch.empty(inp_node, out_node), requires_grad=True)
    torch.nn.init.xavier_normal_(self.weight, gain=1.0)
    torch.clamp(self.weight,-3,3)
    self.fn = MPLayer_in_K.SpikeK.apply

  def forward(self, inputp, inputn=None):
      inputp = torch.unsqueeze(inputp,axis=-1)
      self.weight.type_as(inputp)
      if(inputn==None):
        plusIn =F.relu((3+inputp))
        minusIn =F.relu((3-inputp))
      else:
        minusIn = torch.unsqueeze(inputn,axis=-1)
        plusIn = inputp

      plusW = F.relu(self.weight)
      minusW = F.relu(-self.weight)

      zPlus = torch.cat([(plusIn+plusW),(minusIn+minusW)],axis=1)
      zMinus = torch.cat([(plusIn+minusW),(minusIn+plusW)],axis=1)

      zPlus = self.fn(zPlus, self.gamma)
      zMinus = self.fn(zMinus, self.gamma)
      torch.cuda.empty_cache()
      if(self.diff == 0):
        return zPlus - zMinus  ## previous TEMP based codes will not be compatible because of this change
      else:
        return zPlus,zMinus

  @staticmethod
  class SpikeK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inMat1, gamma):
        ctx.gamma = gamma
        ctx.save_for_backward(inMat1)
        if gamma == 0 or gamma == 1:
            out = torch.kthvalue(inMat1, 1, dim=1).values
            return out
        thr, _ = torch.topk(inMat1, int(gamma), dim=1, largest=False, sorted=False)
        sum_nonzero = thr.sum(dim=1)
        return sum_nonzero / gamma  # average of the smallest gamma values

    @staticmethod
    def backward(ctx, grad_output):
        gamma = ctx.gamma
        inMat1, = ctx.saved_tensors
        grad_input = torch.zeros_like(inMat1)
        thr,indices = torch.topk(inMat1, gamma, dim=1, largest=False, sorted=False)
        grad_vals = (grad_output / gamma).unsqueeze(1).expand_as(indices)
        grad_input.scatter_(dim=1, index=indices, src=grad_vals)
        return grad_input, None

def batch_pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_centered = x - x.mean(dim=1, keepdim=True)
    y_centered = y - y.mean(dim=1, keepdim=True)
    numerator = torch.sum(x_centered * y_centered, dim=1)
    denominator = torch.sqrt(torch.sum(x_centered**2, dim=1)) * torch.sqrt(torch.sum(y_centered**2, dim=1))
    return numerator / (denominator + 1e-6)

# Define 3-layers MLP
import torch
import torch.nn.functional as F
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='Datasets', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='Datasets', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Network parameters
input_dim = 28 * 28  # MNIST images (flattened)
hidden_dim = 800
output_dim = 10
lr = 0.1
epochs = 30
device = "cuda:0"
# Initialize weights (manual)
W1 = torch.randn((hidden_dim, input_dim), device=device) * 0.01
W2 = torch.randn((output_dim, hidden_dim), device=device) * 0.01

# Enable gradient tracking
W1.requires_grad_()
W2.requires_grad_()

# Training loop
for epoch in range(epochs):
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        x = images.view(-1, input_dim).to(device)
        y = labels.to(device)

        # Forward pass
        h = F.relu(F.linear(x, W1))
        logits = F.linear(h, W2)

        loss = F.cross_entropy(logits, y)
        total_loss += loss.item()

        # Backward pass
        loss.backward()

        with torch.no_grad():
            for W in [W1, W2]:
                W -= lr * W.grad
                W.grad.zero_()

        # Accuracy during training
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = correct / total * 100
    print(f"Epoch {epoch+1}: Train Loss = {total_loss/len(train_loader):.4f}, Accuracy = {acc:.2f}%")

# --- Test Evaluation ---
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
W1.requires_grad = False
W2.requires_grad = False
correct_test = 0
total_test = 0

with torch.no_grad():
    for images, labels in test_loader:
        x = images.view(-1, input_dim).to(device)
        y = labels.to(device)

        h_mlp = F.relu(F.linear(x, W1))
        logits_mlp = F.linear(h_mlp, W2)
        preds_mlp = torch.argmax(logits_mlp, dim=1)

        correct_test += (preds_mlp == y).sum().item()
        total_test += y.size(0)

test_acc = correct_test / total_test * 100
print(f"\nTest Accuracy: {test_acc:.2f}%")

print("TOPK inference")
K = [115,75] #90
layer1 = MPLayer_in_K(inp_node=input_dim, out_node=hidden_dim, gamma=K[0], diff=0)
layer1.weight.data = W1.t().clone()
layer2 = MPLayer_in_K(inp_node=hidden_dim, out_node=hidden_dim, gamma=K[1], diff=0)
layer2.weight.data = W2.t().clone()
correct_test = 0
total_test = 0
scaling = 100
with torch.no_grad():
    for images, labels in train_loader:
        x = images.view(-1, input_dim).to(device)
        y = labels.to(device)

        h = F.relu(-layer1(x)*scaling)
        logits = -layer2(h)*scaling
        preds = torch.argmax(logits, dim=1)

        correct_test += (preds == y).sum().item()
        total_test += y.size(0)

test_acc = correct_test / total_test * 100
print(f"\nTrain Accuracy: {test_acc:.2f}%")

#gradient descent on topk networks with scaling
# Define 3-layers MLP
layer11 = MPLayer_in_K(inp_node=input_dim, out_node=hidden_dim, gamma=K[0], diff=0)
layer11.weight.data = W1.t().clone()
layer22 = MPLayer_in_K(inp_node=hidden_dim, out_node=hidden_dim, gamma=K[1], diff=0)
layer22.weight.data = W2.t().clone()
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Network parameters
input_dim = 28 * 28  # MNIST images (flattened)
hidden_dim = 800
output_dim = 10
lr = 0.1
epochs = 20

train_acc_list = []
test_acc_list = []
loss_list = []
optimizer = torch.optim.SGD(list(layer11.parameters()) + list(layer22.parameters()), lr=lr)

# Training loop
scaling=100
for epoch in range(epochs):
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        x = images.view(-1, input_dim).to(device)
        y = labels.to(device)

        h = F.relu(-scaling * layer11(x))
        logits = -scaling * layer22(h)

        loss = F.cross_entropy(logits, y)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_acc = correct / total * 100
    train_acc_list.append(train_acc)
    loss_list.append(total_loss / len(train_loader))

    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in test_loader:
            x = images.view(-1, input_dim).to(device)
            y = labels.to(device)

            h = F.relu(-scaling * layer11(x))
            logits = -scaling * layer22(h)
            preds = torch.argmax(logits, dim=1)

            correct_test += (preds == y).sum().item()
            total_test += y.size(0)

    test_acc = correct_test / total_test * 100
    test_acc_list.append(test_acc)
    print(f"Epoch {epoch+1}: Train Loss = {loss_list[-1]:.4f}, Train Acc = {train_acc:.2f}%, Test Acc = {test_acc:.2f}%")


