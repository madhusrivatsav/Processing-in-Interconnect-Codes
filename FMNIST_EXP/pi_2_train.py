import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from lightning.fabric import Fabric, seed_everything
from torchmetrics.classification import Accuracy
import os
import sys
import time
import numpy as np

# ──────────────────────── Path Configuration ────────────────────────
current_dir = "" ##path to current directory
arch_dir = os.path.join(current_dir, 'network_arch')
result_dir = os.path.join(current_dir, 'Trained_models')

print(current_dir)
print(arch_dir)
print(result_dir)

# Add model definition path
sys.path.append(arch_dir)

# Set device and seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(3407)

# Import custom LeNet5-based architecture with K-top pruning
from LeNet_mnist import LeNet5_K

# ──────────────────────── Loss Function (Distillation) ────────────────────────
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    distill = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    cls = F.cross_entropy(student_logits, labels)
    return alpha * distill + (1 - alpha) * cls

# ──────────────────────── Evaluation Functions ────────────────────────
def evaluatef(model, test_loader):
    """
    Evaluation function without Fabric (used for teacher or base evaluation).
    """
    model.eval()
    test_loss = 0
    accuracy = Accuracy(task="multiclass", num_classes=10).to(device)

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.to(device))
            test_loss += F.cross_entropy(output, target.to(device), reduction="sum").item()
            accuracy(output, target)

    test_loss /= len(test_loader.dataset)
    acc = accuracy.compute().item() * 100
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {acc:.2f}%\n")
    return test_loss, acc

def evaluate(fabric, model, test_loader):
    """
    Evaluation function with Fabric (DDP-compatible).
    """
    model.eval()
    test_loss = 0
    accuracy = Accuracy(task="multiclass", num_classes=10).to(fabric.device)

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            accuracy(output, target)

    test_loss /= len(test_loader.dataset)
    acc = accuracy.compute().item() * 100
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {acc:.2f}%\n")
    return test_loss, acc

# ──────────────────────── Main Function ────────────────────────
def main():
    # Initialize Fabric for distributed and mixed-precision training
    fabric = Fabric(accelerator="cuda", devices=[2,3,4,5], strategy="ddp", precision="16-mixed")
    fabric.launch()

    seed_everything(42)

    # ───── Data Preparation ─────
    transform = transforms.ToTensor()
    PATH_DATASETS = os.path.join(current_dir, "Datasets")

    Fmnist_train = FashionMNIST(PATH_DATASETS, train=True, download=True, transform=transform)
    Fmnist_test = FashionMNIST(PATH_DATASETS, train=False, download=True, transform=transform)

    # Split training data: 90% train, 10% validation
    val_size = int(0.1 * len(Fmnist_train))
    train_size = len(Fmnist_train) - val_size
    train_data, val_data = random_split(Fmnist_train, [train_size, val_size])

    batch_size = 128
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(Fmnist_test, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)

    # Setup Fabric for dataloaders
    train_loader, test_loader, val_loader = fabric.setup_dataloaders(train_loader, test_loader, val_loader)

    # ───── Define Student Network (K-top pruning applied) ─────
    K = [12, 50, 85, 25, 75]  # Top-K values per layer
    model = LeNet5_K(gamma=K, temp1=True, temp2=True, temp3=True, temp4=True, temp5=True).to(device)

    # ───── Load Pre-trained Teacher Model ─────
    model_T = LeNet5_K().to(device)
    model_T.load_state_dict(torch.load(os.path.join(result_dir, "FMNIST/fmnist_lenet5.pt")))

    # ───── Copy compatible weights from teacher to student ─────
    for p_model, p_model_T in zip(model.parameters(), model_T.parameters()):
        if p_model.size() == p_model_T.size():
            p_model.data = p_model_T.data.contiguous().clone()
        else:
            print("hi")
            p_model.data = p_model_T.data.view(p_model_T.size()[0], -1).T

    # ───── Evaluate Initial Accuracy ─────
    evaluatef(model, test_loader)
    evaluatef(model_T, test_loader)

    # ───── Optimizer and Scheduler ─────
    epochs = 20
    lr = 5e-5
    optimizer = optim.Adamax(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    model, optimizer = fabric.setup(model, optimizer)
    log_path = os.path.join(result_dir, "FMNIST/fmnist_training_wt_dist_KD")
     # Optional: Write header to file
    with open(log_path, 'w') as f:
            f.write("epoch,train_loss,test_accuracy\n")
    train_losses = []
    test_accuracies = []
    evaluate(fabric, model, test_loader)

    # ───── Training Loop ─────
    print('Timing started...')
    start = time.time()
    accmax = 0
    train_losses = []
    temperature = 0
    alpha = 0.5
    
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        num_batches = 0
        print(f"\nEpoch {epoch}/{epochs}")
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for data, target in train_loader:
            with torch.no_grad():
                model_T = model_T.to(fabric.device)
                teacher_output = model_T(data)

            student_output = model(data)
            loss = distillation_loss(student_output, teacher_output, target, temperature, alpha)
            # output = model(data)
            # loss = F.cross_entropy(output, target)
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1
            running_loss += loss.item()
            num_batches += 1

        # Log average training loss
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        print(f"Average Training Loss: {avg_loss:.4f}")

        # Evaluate on test set
        _, acc = evaluate(fabric, model, test_loader)
        scheduler.step()
        avg_train_loss = running_loss / num_batches
        train_losses.append(avg_train_loss)
        test_accuracies.append(acc)
        with open(log_path, 'a') as f:
         f.write(f"{epoch},{avg_train_loss:.4f},{acc:.2f}\n")
        # Save best-performing model
        if acc > accmax:
            accmax = acc
            torch.save(model.state_dict(), os.path.join(result_dir, "FMNIST/fmnist_kk_5.pt"))
            print(f"✔ Model saved with best accuracy: {acc:.2f}%")

    # ───── Final Evaluation ─────
    end = time.time()
    print(f"Training completed in {(end - start)/60:.2f} minutes")
    evaluate(fabric, model, test_loader)
    # Peak memory usage
    max_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    print(f"Peak GPU memory usage: {max_memory_mb:.2f} MB")

# ──────────────────────── Entry Point ────────────────────────
if __name__ == "__main__":
    main()
