import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as tt
from lightning.fabric import Fabric, seed_everything
from torchmetrics.classification import Accuracy

# ───────────── CLI ARGUMENT PARSER ─────────────
parser = argparse.ArgumentParser(description="Train ResNet9 on CIFAR-100 with Lightning Fabric")
parser.add_argument("--data_dir", type=str, required=True, help="Path to CIFAR-100 dataset")
parser.add_argument("--save_path", type=str, required=True, help="Path to save trained model")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
parser.add_argument("--device_ids", type=int, nargs='+', default=[0], help="List of CUDA device IDs")
parser.add_argument("--seed", type=int, default=None, help="Manual seed for weight initialization (optional)")
args = parser.parse_args()

# ───────────── PATH SETUP ─────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
arch_dir = os.path.join(current_dir, 'network_arch')
sys.path.append(arch_dir)
from resnet_9 import ResNet9_100

# ───────────── Evaluation Function ─────────────
def evaluate(fabric, model, dataloader):
    model.eval()
    loss = 0
    acc = Accuracy(task="multiclass", num_classes=100).to(fabric.device)
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            loss += F.cross_entropy(output, target, reduction="sum").item()
            acc.update(output, target)
    avg_loss = loss / len(dataloader.dataset)
    accuracy = acc.compute().item() * 100
    print(f"\nTest set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n")
    return avg_loss, accuracy

# ───────────── Optional Weight Initialization ─────────────
def initialize_weights_with_seed(model, seed):
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.kaiming_normal_(m.weight)

# ───────────── Main ─────────────
def main():
    fabric = Fabric(accelerator="cuda", devices=args.device_ids, strategy="ddp", precision="16-mixed")
    fabric.launch()
    seed_everything(42)

    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    transform_train = tt.Compose([
        tt.RandomCrop(32, padding=4, padding_mode='reflect'),
        tt.RandomHorizontalFlip(),
        tt.ToTensor(),
        tt.Normalize(*stats, inplace=True)
    ])
    transform_test = tt.Compose([
        tt.ToTensor(),
        tt.Normalize(*stats)
    ])

    train_dataset = CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
    test_dataset = CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    model = ResNet9_100(3, out=100)

    if args.seed is not None:
        initialize_weights_with_seed(model, args.seed)
        print(f"Initialized weights with seed {args.seed}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        three_phase=True
    )

    model, optimizer = fabric.setup(model, optimizer)

    print("Starting training...")
    start = time.time()
    evaluate(fabric, model, test_loader)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
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

    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved at {args.save_path}")
    print(f"Total time elapsed: {(time.time() - start) / 60:.2f} minutes")

if __name__ == "__main__":
    main()
