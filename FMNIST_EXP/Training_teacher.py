import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from lightning.fabric import Fabric, seed_everything
from torchmetrics.classification import Accuracy
import os
import sys
import time

# ────────────────────── Setup and Paths ──────────────────────
current_dir = "/home/madhu/TEMP_FINAL_CODES"
arch_dir = os.path.join(current_dir, 'network_arch')
result_dir = os.path.join(current_dir, 'Trained_models')

sys.path.append(arch_dir)  # Add custom model path to sys
print(f"Arch dir: {arch_dir}")
print(f"Result dir: {result_dir}")

# ────────────────────── Model Import ──────────────────────
from LeNet_mnist import LeNet5_K  # Custom LeNet variant for FMNIST

# ────────────────────── Evaluation Function ──────────────────────
def evaluate(fabric, model, dataloader):
    """
    Evaluate model accuracy and loss on test/val data.
    """
    model.eval()
    test_loss = 0.0
    accuracy = Accuracy(task="multiclass", num_classes=10).to(fabric.device)

    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            accuracy.update(output, target)

    avg_loss = test_loss / len(dataloader.dataset)
    acc = accuracy.compute().item() * 100
    print(f"\nTest Set → Avg Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
    return avg_loss, acc

# ────────────────────── Main Training Routine ──────────────────────
def main():
    # Fabric setup (DDP + AMP)
    fabric = Fabric(accelerator="cuda", devices=[2,3,4,5], strategy="ddp", precision="16-mixed")
    fabric.launch()
    seed_everything(42)

    # ───── Data loading & transformation ─────
    transform = transforms.ToTensor()
    dataset_path = os.path.join(current_dir, 'Datasets')

    # Load FashionMNIST
    full_train = FashionMNIST(dataset_path, train=True, download=True, transform=transform)
    test_data = FashionMNIST(dataset_path, train=False, download=True, transform=transform)

    # Split into 90% train and 10% validation
    train_data, val_data = random_split(full_train, [54000, 6000])

    # DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)

    # Apply Fabric dataloader setup
    train_loader, test_loader, val_loader = fabric.setup_dataloaders(train_loader, test_loader, val_loader)

    # ───── Model, optimizer, and scheduler ─────
    model = LeNet5_K()
    model.to(fabric.device)

    optimizer = optim.Adamax(model.parameters(), lr=5e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # Apply Fabric model and optimizer setup
    model, optimizer = fabric.setup(model, optimizer)

    # ───── Training loop ─────
    epochs = 20
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        print(f"\nEpoch {epoch}/{epochs} — LR: {optimizer.param_groups[0]['lr']:.5f}")

        for data, target in train_loader:
            output = model(data)
            loss = F.cross_entropy(output, target)
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        scheduler.step()

        # Evaluate and save best model
        _, acc = evaluate(fabric, model, test_loader)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(result_dir, "fmnist_lenet5.pt"))
            print(f"✔ Best model saved with accuracy: {best_acc:.2f}%")

    # ───── Final reporting ─────
    total_time = (time.time() - start_time) / 60
    print(f"\nTraining completed in {total_time:.2f} minutes")
    evaluate(fabric, model, test_loader)

# ────────────────────── Entry Point ──────────────────────
if __name__ == "__main__":
    main()
