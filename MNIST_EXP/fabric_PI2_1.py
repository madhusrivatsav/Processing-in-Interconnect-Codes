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

# Set directory paths
current_dir = "" ## set the path of your current directory
arch_dir = os.path.join(current_dir, "network_arch")
result_dir = os.path.join(current_dir, "Trained_models")
sys.path.append(arch_dir)

print("Code Directory:", current_dir)
print("Architecture Directory:", arch_dir)
print("Model Save Directory:", result_dir)

# Set random seed and device
torch.manual_seed(3407)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import custom TEMP network architectures
from PI2_MLP import TEMP_Network_hybrid_nobn

def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """
    Knowledge distillation loss combining soft targets from teacher and hard targets from ground truth.
    """
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    distill = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature**2)
    classification = F.cross_entropy(student_logits, labels)
    return alpha * distill + (1 - alpha) * classification

def evaluate(fabric, model, dataloader):
    """
    Evaluate the model's average loss and accuracy over a given dataset.
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
    print(f"\nTest set: Average loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%\n")
    return avg_loss, acc

def main():
    # Initialize Fabric for multi-GPU mixed precision training
    fabric = Fabric(accelerator="cuda", devices=[2, 3, 4, 5], strategy="ddp", precision="16-mixed")
    fabric.launch()
    seed_everything(42)

    # Data transformations and loading
    transform = tt.ToTensor()
    data_dir = os.path.join(current_dir, "Datasets")
    mnist_train = MNIST(data_dir, train=True, download=True, transform=transform)
    mnist_test = MNIST(data_dir, train=False, download=True, transform=transform)

    batch_size = 32
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)

    # Setup Fabric-managed dataloaders
    train_loader, test_loader, val_loader = fabric.setup_dataloaders(train_loader, test_loader, val_loader)

    # Load student model with K=1 constraint and pretrained weights
    gamma = [1, 1]
    hidden_dim = 500
    student_model = TEMP_Network_hybrid_nobn(in_features=784, num_layers=2, num_hidden_list=[hidden_dim, 10], gamma=gamma)
    student_ckpt = os.path.join(result_dir, "MNIST/mnist_kk_500.pt")
    student_model.load_state_dict(torch.load(student_ckpt))

    # Load teacher model with pretrained weights
    teacher_model = TEMP_Network_hybrid_nobn(in_features=784, num_layers=2, num_hidden_list=[hidden_dim, 10])
    teacher_ckpt = os.path.join(result_dir, "MNIST/mnist_mlp_500.pt")
    teacher_model.load_state_dict(torch.load(teacher_ckpt))

    # Optimizer setup
    optimizer = optim.Adamax(student_model.parameters(), lr=1e-2)
    student_model, optimizer = fabric.setup(student_model, optimizer)

    # Distillation hyperparameters
    temperature = 5.0
    alpha = 0.5
    epochs = 20
    best_acc = 0

    print("Starting distillation training...\n")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        print("Current LR:", optimizer.param_groups[0]['lr'])

        student_model.train()
        teacher_model = teacher_model.to(fabric.device)

        for batch_idx, (data, target) in enumerate(train_loader):
            # Forward pass
            student_output = student_model(data)
            with torch.no_grad():
                teacher_output = teacher_model(data)

            # Compute distillation loss
            loss = distillation_loss(student_output, teacher_output, target, temperature, alpha)

            # Backward pass and optimization
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        # Validation evaluation
        _, val_acc = evaluate(fabric, student_model, val_loader)
        print("Test Evaluation:")
        _, test_acc = evaluate(fabric, student_model, test_loader)

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(result_dir, "MNIST/mnist_kk_500_1.pt")
            torch.save(student_model.state_dict(), save_path)

    # Training summary
    elapsed_time = (time.time() - start_time) / 60
    print(f"Total training time: {elapsed_time:.2f} minutes")

    print("Final Evaluation on Test Set:")
    evaluate(fabric, student_model, test_loader)

if __name__ == "__main__":
    main()
