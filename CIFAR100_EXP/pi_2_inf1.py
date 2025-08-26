import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import  CIFAR100
from lightning.fabric import Fabric, seed_everything
from torchmetrics.classification import Accuracy
import os
import sys
import time
import numpy as np

current_dir = "/home/madhu/TEMP_FINAL_CODES"
print(current_dir)
arch_dir = os.path.join(current_dir, 'network_arch')
sys.path.append(arch_dir)
print(arch_dir)
result_dir = os.path.join(current_dir, 'Trained_models')
print(result_dir)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(3407)
from resnet_9 import ResNet9_100_temp_noise
import numpy as np


# Function to register hooks and store gradients
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    # Soft targets from teacher and student
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_probs = F.log_softmax(student_logits / temperature, dim=1)
    
    # Distillation loss (soft cross-entropy)
    distillation_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature**2)
    
    # Classification loss (hard cross-entropy)
    classification_loss = F.cross_entropy(student_logits, labels)
    
    # Combined loss
    return alpha * distillation_loss + (1 - alpha) * classification_loss

def evaluate(fabric, model, test_loader):
    model.eval()
    test_loss = 0
    accuracy = Accuracy(task="multiclass", num_classes=100).to(fabric.device)

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            accuracy(output, target)

    test_loss /= len(test_loader.dataset)
    acc = accuracy.compute().item() * 100

    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {acc:.2f}%\n")
    return test_loss, acc

def main():
    fabric = Fabric(accelerator="cuda", devices=4, strategy="ddp", precision="16-mixed")
    fabric.launch()

    seed_everything(60)
    import torchvision.transforms as tt
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    train_transform = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         tt.RandomHorizontalFlip(), 
                         tt.ToTensor(), 
                         tt.Normalize(*stats,inplace=True)])
    test_transform = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])
    PATH_DATASETS = "/home/madhu/TEMP_FINAL_CODES/Datasets"
   
    train_dataset = CIFAR100(PATH_DATASETS, train=True, download=True,transform=train_transform)
    test_dataset = CIFAR100(PATH_DATASETS, train=False, download=True, transform=test_transform)
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)
    
    # device = "cuda"
    #student network
    gamma = [10,50,55,65,100,100,80,250,50] #(16,50,55,65,60,80,90,200,50) #
    # model = ResNet9_100_temp1(3,out=100,gamma=gamma,temp0=True,temp1=True,temp2=True,temp3=True,temp4=True,temp5=True,temp6=True,temp7=True,temp8=True)
    model = ResNet9_100_temp_noise(3,out=100,gamma=gamma,temp0=True,temp1=True,temp2=True,temp3=True,temp4=True,temp5=True,temp6=True,temp7=True,temp8=True)
    model = model.to(device)
    model.load_state_dict(torch.load("/home/madhu/TEMP_FINAL_CODES/CIFAR100/Trained_models/ResNet9_cifar100_k_new_new.pt"))


    epochs = 60
    lr = 1e-2
    accmax = 0
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-7) #changing the wt decay from 1e-4 (89.66)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs,steps_per_epoch=len(train_loader),pct_start=0.35) #0.4
    model, optimizer = fabric.setup(model, optimizer)
    start = time.time()
    print('timing started')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    evaluate(fabric, model, test_loader)



if __name__ == "__main__":
    main()
    
 