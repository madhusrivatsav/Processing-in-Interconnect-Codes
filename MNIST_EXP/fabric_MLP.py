import os
import sys
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as tt
from lightning.fabric import Fabric, seed_everything
from torchmetrics.classification import Accuracy

# Path setup
current_dir = "" ## specify the current directory path here
arch_dir = os.path.join(current_dir, "network_arch")
result_dir = os.path.join(current_dir, "Trained_models")
sys.path.append(arch_dir)

print("Code Directory:", current_dir)
print("Architecture Directory:", arch_dir)
print("Model Save Directory:", result_dir)

# Set device and seed
torch.manual_seed(3407)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import TEMP model
from TEMP_MLP import TEMP_Network_hybrid_nobn

def evaluate(fabric, model, dataloader):
    """
    Evaluate the model on the given dataset and return average loss and accuracy.
    """
    model.eval()
    total_loss = 0
    accuracy = Accuracy(task="multiclass", num_classes=10).to(fabric.device)

    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            total_loss += F.cross_entropy(output, target, reduction="sum").item()
            accuracy.update(output, target)

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy.compute().item() * 100
    print(f"\nEvaluation - Avg Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%\n")
    return avg_loss, acc

def main():
    # Initialize Fabric for multi-GPU training with mixed precision
    fabric = Fabric(accelerator="cuda", devices=[4, 5], strategy="ddp", precision="16-mixed")
    fabric.launch()
    seed_everything(42)

    # Define transforms
    transform = tt.ToTensor()
    data_dir = os.path.join(current_dir, "Datasets")

    # Load MNIST dataset
    mnist_train = MNIST(data_dir, train=True, download=True, transform=transform)
    mnist_test = MNIST(data_dir, train=False, download=True, transform=transform)

    # Define data loaders
    batch_size = 32
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)

    # Setup dataloaders with Fabric
    train_loader, test_loader, val_loader = fabric.setup_dataloaders(train_loader, test_loader, val_loader)

    # Initialize TEMP model (no batch norm)
    model = TEMP_Network_hybrid_nobn(in_features=784, num_layers=2, num_hidden_list=[100, 10])

    # Training hyperparameters
    epochs = 30
    lr = 5e-3
    optimizer = optim.Adamax(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, min_lr=1e-6)

    # Prepare model and optimizer for Fabric
    model, optimizer = fabric.setup(model, optimizer)

    print("Training started...\n")
    start = time.time()

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            output = model(data)
            loss = F.cross_entropy(output, target)
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        # Validation accuracy and scheduler step
        val_loss, val_acc = evaluate(fabric, model, val_loader)
        scheduler.step(val_acc)

        # Evaluate on test set
        print("Test Evaluation:")
        evaluate(fabric, model, test_loader)

    elapsed = (time.time() - start) / 60
    print(f"Total training time: {elapsed:.2f} minutes")

    # Final test evaluation
    print("Final Test Evaluation:")
    evaluate(fabric, model, test_loader)

    # Save trained model
    save_path = os.path.join(result_dir, "mnist_mlp_100.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
