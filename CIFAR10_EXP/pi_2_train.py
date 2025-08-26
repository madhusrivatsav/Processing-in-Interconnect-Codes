import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as tt
from lightning.fabric import Fabric, seed_everything
from torchmetrics.classification import Accuracy

# ──────────────────────── Argument Parser ─────────────────────────
parser = argparse.ArgumentParser(description="Knowledge Distillation on CIFAR10 with ResNet9")
parser.add_argument("--current_dir", type=str, required=True, help="Path to TEMP")
parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
parser.add_argument("--teacher_ckpt", type=str, required=True, help="Path to teacher model checkpoint")
parser.add_argument("--save_path", type=str, required=True, help="Path to save distilled student model")
parser.add_argument("--batch_size", type=int, default=28, help="Batch size for training")
parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
args = parser.parse_args()

# ──────────────────────── Setup ─────────────────────────
current_dir = args.current_dir
arch_dir = os.path.join(current_dir, "network_arch")
sys.path.append(arch_dir)

from resnet_9 import ResNet9_100

torch.manual_seed(3407)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ──────────────────────── Cutout Augmentation ─────────────────────────
class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y, x = np.random.randint(h), np.random.randint(w)
            y1, y2 = np.clip([y - self.length // 2, y + self.length // 2], 0, h)
            x1, x2 = np.clip([x - self.length // 2, x + self.length // 2], 0, w)
            mask[y1:y2, x1:x2] = 0.
        return img * torch.from_numpy(mask).expand_as(img)

# ──────────────────────── Loss Function ─────────────────────────
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    distill = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    cls = F.cross_entropy(student_logits, labels)
    return alpha * distill + (1 - alpha) * cls

# ──────────────────────── Evaluation ─────────────────────────
def evaluate(fabric, model, dataloader):
    model.eval()
    loss = 0
    accuracy = Accuracy(task="multiclass", num_classes=10).to(fabric.device)
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            loss += F.nll_loss(output, target, reduction="sum").item()
            accuracy.update(output, target)
    avg_loss = loss / len(dataloader.dataset)
    acc = accuracy.compute().item() * 100
    print(f"\nTest set: Average loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%\n")
    return avg_loss, acc

# ──────────────────────── Main Training Function ─────────────────────────
def main():
    fabric = Fabric(accelerator="cuda", devices=[2,3,4,5], strategy="ddp", precision="16-mixed")
    fabric.launch()

    seed_everything(42)

    # Data augmentation
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = tt.Compose([
        tt.RandomCrop(32, padding=4, padding_mode='reflect'),
        tt.RandomHorizontalFlip(),
        tt.ToTensor(),
        Cutout(n_holes=1, length=16),
        tt.Normalize(*stats, inplace=True)
    ])
    test_transform = tt.Compose([
        tt.ToTensor(),
        tt.Normalize(*stats)
    ])

    train_dataset = CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(args.data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    # Define models
    gamma = [16, 50, 35, 35, 60, 70, 80, 200, 50]
    student = ResNet9_100(3, out=10, gamma=gamma,
        temp0=True, temp1=True, temp2=True, temp3=True,
        temp4=True, temp5=True, temp6=True, temp7=True, temp8=True).to(device)

    teacher = ResNet9_100(3, out=10)
    teacher.load_state_dict(torch.load(args.teacher_ckpt))
    teacher = teacher.to(device)

    # Optionally copy compatible weights
    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
        if p_s.size() == p_t.size():
            p_s.data = p_t.data.clone()
        else:
            reshaped = p_t.data.clone().contiguous().view(p_t.size()[0], -1).T
            p_s.data = reshaped.contiguous()

    # Optimizer and Scheduler
    optimizer = optim.Adam(student.parameters(), lr=1e-2, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-2, epochs=args.epochs, steps_per_epoch=len(train_loader),
        three_phase=True, final_div_factor=10.0, pct_start=0.4
    )

    student, optimizer = fabric.setup(student, optimizer)
    temperature = 5.0
    alpha = 0.5
    accmax = 0

    print(f"Model Parameters: {sum(p.numel() for p in student.parameters())}")
    print("Initial Test Accuracy:")
    evaluate(fabric, student, test_loader)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        student.train()
        for data, target in train_loader:
            with torch.no_grad():
                teacher_output = teacher(data)
            student_output = student(data)
            loss = distillation_loss(student_output, teacher_output, target, temperature, alpha)
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        _, acc = evaluate(fabric, student, test_loader)
        if acc > accmax:
            accmax = acc
            torch.save(student.state_dict(), args.save_path)

    elapsed = (time.time() - start_time) / 60
    print(f"Training completed in {elapsed:.2f} minutes")
    # Peak memory usage
    max_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    print(f"Peak GPU memory usage: {max_memory_mb:.2f} MB")
    
if __name__ == "__main__":
    main()
