import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as tt
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from lightning.fabric import Fabric, seed_everything
from torchmetrics.classification import Accuracy

# Paths setup
current_dir = "" ##specify path of your current directory 
arch_dir = os.path.join(current_dir, 'network_arch')
result_dir = os.path.join(current_dir, 'Trained_models')
sys.path.append(arch_dir)

print("Code Directory:", current_dir)
print("Architecture Directory:", arch_dir)
print("Model Save Directory:", result_dir)

# Set device and seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(3407)

# Import custom network definitions
from PI2_MLP import TEMP_Network_hybrid_nobn

def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """
    Compute the knowledge distillation loss combining soft and hard targets.
    """
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    cls_loss = F.cross_entropy(student_logits, labels)
    return alpha * distill_loss + (1 - alpha) * cls_loss

def evaluate(fabric, model, test_loader):
    """
    Evaluate the model on test/validation data and return average loss and accuracy.
    """
    model.eval()
    test_loss = 0
    accuracy = Accuracy(task="multiclass", num_classes=10).to(fabric.device)

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            accuracy.update(output, target)

    test_loss /= len(test_loader.dataset)
    acc = accuracy.compute().item() * 100
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {acc:.2f}%\n")
    return test_loss, acc

def main():
    # Initialize Fabric for multi-GPU training with mixed precision
    fabric = Fabric(accelerator="cuda", devices=[2, 3, 4, 5], strategy="ddp", precision="16-mixed")
    fabric.launch()

    # Reproducibility
    seed_everything(42)

    # Dataset setup
    data_path = os.path.join(current_dir, "Datasets")
    transform = tt.ToTensor()
    mnist_train = MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = MNIST(data_path, train=False, download=True, transform=transform)

    batch_size = 32
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)

    # Distribute data loaders using Fabric
    train_loader, test_loader, val_loader = fabric.setup_dataloaders(train_loader, test_loader, val_loader)

    # Model instantiation
    K = [1, 30]
    hidden_dim = 500

    # Student model (no batch norm)
    student_model = TEMP_Network_hybrid_nobn(in_features=784, num_layers=2,
                                             num_hidden_list=[hidden_dim, 10], gamma=K)
    # Teacher model (with batch norm)
    teacher_model = TEMP_Network_hybrid_nobn(in_features=784, num_layers=2,
                                        num_hidden_list=[hidden_dim, 10])
    teacher_ckpt = os.path.join(result_dir, "MNIST/mnist_mlp_500.pt")
    teacher_model.load_state_dict(torch.load(teacher_ckpt))

    # Copy teacher weights into student model (reshape if needed)
    for p_student, p_teacher in zip(student_model.parameters(), teacher_model.parameters()):
        if p_student.size() == p_teacher.size():
            p_student.data = p_teacher.data.clone()
        else:
            reshaped = p_teacher.data.view(p_teacher.size(0), -1).T
            p_student.data = reshaped.clone()

    # Optimizer setup
    optimizer = optim.Adamax(student_model.parameters(), lr=1e-2)
    student_model, optimizer = fabric.setup(student_model, optimizer)

    temperature = 5.0
    alpha = 0.5
    epochs = 20
    best_acc = 0

    start_time = time.time()
    print("Training started...\n")

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        print("Current LR:", optimizer.param_groups[0]['lr'])

        student_model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Get student and teacher outputs
            output_student = student_model(data)
            teacher_model = teacher_model.to(fabric.device)
            with torch.no_grad():
                output_teacher = teacher_model(data)

            # Compute distillation loss
            loss = distillation_loss(output_student, output_teacher, target, temperature, alpha)

            # Backward pass
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        # Validation and test evaluation
        val_loss, val_acc = evaluate(fabric, student_model, val_loader)
        test_loss, test_acc = evaluate(fabric, student_model, test_loader)

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(result_dir, "MNIST/mnist_kk_500.pt")
            torch.save(student_model.state_dict(), save_path)

    total_time = (time.time() - start_time) / 60
    print(f"Total training time: {total_time:.2f} minutes")
    evaluate(fabric, student_model, test_loader)

if __name__ == "__main__":
    main()
