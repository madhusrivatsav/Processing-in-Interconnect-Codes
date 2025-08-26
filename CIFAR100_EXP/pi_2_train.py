import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as tt
from lightning.fabric import Fabric, seed_everything
from torchmetrics.classification import Accuracy

# ───────────── Argument Parser ─────────────
parser = argparse.ArgumentParser(description="Knowledge Distillation for CIFAR-100 using ResNet9")
parser.add_argument("--data_dir", type=str, required=True, help="Path to CIFAR-100 dataset")
parser.add_argument("--teacher_ckpt", type=str, required=True, help="Path to teacher checkpoint")
parser.add_argument("--save_path", type=str, required=True, help="Path to save best student checkpoint")
parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
parser.add_argument("--gamma", type=int, nargs=9, default=[16, 50, 55, 65, 60, 80, 90, 200, 50], help="List of 9 gamma values")
parser.add_argument("--device_ids", type=int, nargs='+', default=[0], help="CUDA device IDs")
args = parser.parse_args()

# ───────────── Setup Import Path ─────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
arch_dir = os.path.join(current_dir, "network_arch")
sys.path.append(arch_dir)
from resnet_9 import ResNet9_100

# ───────────── Distillation Loss ─────────────
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_probs = F.log_softmax(student_logits / temperature, dim=1)
    distill = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    cls = F.cross_entropy(student_logits, labels)
    return alpha * distill + (1 - alpha) * cls

# ───────────── Evaluation ─────────────
def evaluate(fabric, model, dataloader):
    model.eval()
    loss = 0
    accuracy = Accuracy(task="multiclass", num_classes=100).to(fabric.device)
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            loss += F.nll_loss(output, target, reduction="sum").item()
            accuracy.update(output, target)
    avg_loss = loss / len(dataloader.dataset)
    acc = accuracy.compute().item() * 100
    print(f"\nTest set: Avg loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%\n")
    return avg_loss, acc

# ───────────── Main ─────────────
def main():
    fabric = Fabric(accelerator="cuda", devices=args.device_ids, strategy="ddp", precision="16-mixed")
    fabric.launch()
    seed_everything(42)

    # Transforms for CIFAR-100
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
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

    # Datasets
    train_dataset = CIFAR100(args.data_dir, train=True, download=True, transform=train_transform)
    test_dataset = CIFAR100(args.data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    # Models
    student = ResNet9_100(3, out=100, gamma=args.gamma,
        temp0=True, temp1=True, temp2=True, temp3=True,
        temp4=True, temp5=True, temp6=True, temp7=True, temp8=True
    ).to(fabric.device)

    teacher = ResNet9_100(3, out=100)
    teacher.load_state_dict(torch.load(args.teacher_ckpt))
    teacher = teacher.to(fabric.device)

    # Initialize student weights with teacher
    for s_param, t_param in zip(student.parameters(), teacher.parameters()):
        if s_param.size() == t_param.size():
            s_param.data = t_param.data.clone()
        else:
            reshaped = t_param.data.view(t_param.size(0), -1).T
            s_param.data = reshaped.clone()

    optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.
    )

    student, optimizer = fabric.setup(student, optimizer)

    print(f"Total parameters: {sum(p.numel() for p in student.parameters())}")
    temperature = 5.0
    alpha = 0.5
    best_acc = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        student.train()
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        for data, target in train_loader:
            student_output = student(data)
            with torch.no_grad():
                teacher_output = teacher(data)
            loss = distillation_loss(student_output, teacher_output, target, temperature, alpha)
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        _, acc = evaluate(fabric, student, test_loader)
        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), args.save_path)
            print(f"New best model saved with accuracy: {best_acc:.2f}%")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed / 60:.2f} minutes.")

if __name__ == "__main__":
    main()
