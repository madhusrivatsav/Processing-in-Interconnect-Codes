import os
import sys
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
import torchvision.transforms as tt
from lightning.fabric import Fabric, seed_everything
from torchmetrics.classification import Accuracy

# ────────────── Path Setup ──────────────
current_dir = "/home/madhu/TEMP_FINAL_CODES"
arch_dir = os.path.join(current_dir, 'network_arch')
result_dir = os.path.join(current_dir, 'Trained_models')
sys.path.append(arch_dir)
print(current_dir)
print(arch_dir)
print(result_dir)

# ────────────── Device & Seed ──────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(3407)

# ────────────── Model Import ──────────────
from LeNet_mnist import LeNet5_K

# ────────────── Distillation Loss Function ──────────────
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_probs = F.log_softmax(student_logits / temperature, dim=1)
    distill_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    cls_loss = F.cross_entropy(student_logits, labels)
    return alpha * distill_loss + (1 - alpha) * cls_loss

# ────────────── Evaluation Function ──────────────
def evaluate(fabric, model, dataloader):
    model.eval()
    test_loss = 0
    accuracy = Accuracy(task="multiclass", num_classes=10).to(fabric.device)
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            accuracy(output, target)
    avg_loss = test_loss / len(dataloader.dataset)
    acc = accuracy.compute().item() * 100
    print(f"\nEvaluation - Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%\n")
    return avg_loss, acc

# ────────────── Main Training Loop ──────────────
def main():
    # Initialize distributed training context
    fabric = Fabric(accelerator="cuda", devices=[2,3,4,5], strategy="ddp", precision="16-mixed")
    fabric.launch()
    seed_everything(42)

    # ─────── Data Preprocessing ───────
    transform = tt.Compose([tt.ToTensor()])
    dataset_path = os.path.join(current_dir, "Datasets")

    Fmnist_train = FashionMNIST(dataset_path, train=True, download=True, transform=transform)
    Fmnist_test = FashionMNIST(dataset_path, train=False, download=True, transform=transform)

    # Split training set into train/val
    val_size = int(0.1 * len(Fmnist_train))
    train_size = len(Fmnist_train) - val_size
    train_data, val_data = random_split(Fmnist_train, [train_size, val_size])

    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(Fmnist_test, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    train_loader, test_loader, val_loader = fabric.setup_dataloaders(train_loader, test_loader, val_loader)

    # ─────── Load Student Model ───────
    gamma = [10, 10, 10, 10, 10]
    model = LeNet5_K(gamma=gamma, temp1=True, temp2=True, temp3=True, temp4=True, temp5=True)
    model.load_state_dict(torch.load(os.path.join(result_dir, "FMNIST/fmnist_kk_topk1.pt")))

    # ─────── Load Teacher Model ───────
    model_T = LeNet5_K()
    model_T.load_state_dict(torch.load(os.path.join(result_dir, "FMNIST/fmnist_lenet5.pt")))

    # ─────── Optimization Setup ───────
    epochs = 20
    lr = 5e-3
    optimizer = optim.Adamax(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=0.5, min_lr=1e-6)

    # ─────── Distributed Setup ───────
    model, optimizer = fabric.setup(model, optimizer)

    # ─────── Training Loop ───────
    print("Starting training...")
    start = time.time()
    acc_max = 0
    temperature = 5.0
    alpha = 0.5

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        model.train()
        
        for data, target in train_loader:
            with torch.no_grad():
                model_T = model_T.to(fabric.device)
                teacher_output = model_T(data)

            student_output = model(data)
            loss = distillation_loss(student_output, teacher_output, target, temperature, alpha)

            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        # Evaluate on validation and test sets
        val_loss, val_acc = evaluate(fabric, model, val_loader)
        scheduler.step(val_acc)

        print("Test Accuracy:")
        _, test_acc = evaluate(fabric, model, test_loader)

        # Save best model
        if test_acc > acc_max:
            acc_max = test_acc
            torch.save(model.state_dict(),  os.path.join(result_dir, "FMNIST/fmnist_kk_topk10.pt"))

    elapsed = (time.time() - start) / 60
    print(f"\nTraining completed in {elapsed:.2f} minutes")
    evaluate(fabric, model, test_loader)

if __name__ == "__main__":
    main()
