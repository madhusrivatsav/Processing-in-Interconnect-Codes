import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as tt
from lightning.fabric import Fabric, seed_everything
from torchmetrics.classification import Accuracy

# ───────────── Argument Parser ─────────────
parser = argparse.ArgumentParser(description="Train ResNet9 on CIFAR-10")
parser.add_argument("--data_dir", type=str, required=True, help="Path to CIFAR-10 dataset")
parser.add_argument("--save_path", type=str, required=True, help="Path to save trained model (.pt)")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
parser.add_argument("--device_ids", type=int, nargs='+', default=[0], help="CUDA device IDs")
args = parser.parse_args()

# ───────────── Dynamic Import Setup ─────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
arch_dir = os.path.join(current_dir, "network_arch")
sys.path.append(arch_dir)
from resnet_9 import ResNet9_100

# ───────────── Evaluation Function ─────────────
def evaluate(fabric, model, dataloader):
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
    print(f"\nTest set: Avg loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%\n")
    return avg_loss, acc

# ───────────── Main ─────────────
def main():
    fabric = Fabric(accelerator="cuda", devices=args.device_ids, strategy="ddp", precision="16-mixed")
    fabric.launch()
    seed_everything(42)

    # Transforms
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = tt.Compose([
        tt.RandomCrop(32, padding=4, padding_mode='reflect'),
        tt.RandomHorizontalFlip(),
        tt.ToTensor(),
        tt.Normalize(*stats, inplace=True)
    ])
    test_transform = tt.Compose([
        tt.ToTensor(),
        tt.Normalize(*stats)
    ])

    # Dataset
    train_dataset = CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(args.data_dir, train=False, download=True, transform=test_transform)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    # Model
    model = ResNet9_100(in_channels=3, out=10)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        three_phase=True,
        final_div_factor=10.0,
        pct_start=0.3
    )

    model, optimizer = fabric.setup(model, optimizer)

    print("Training started...")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        model.train()
        for data, target in train_loader:
            output = model(data)
            loss = F.cross_entropy(output, target)
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        evaluate(fabric, model, test_loader)

    end = time.time()
    print(f"Training completed in {(end - start)/60:.2f} minutes")
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    main()
