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

# ───────────── Argument Parser ─────────────
parser = argparse.ArgumentParser(description="Evaluate quantized ResNet9 with temp noise on CIFAR10")
parser.add_argument("--current_dir", type=str, required=True, help="Path to dataset directory")
parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to student model checkpoint (.pt)")
parser.add_argument("--n_bits", type=int, default=4, help="Number of bits for symmetric quantization")
parser.add_argument("--std", type=float, default=0.0, help="Standard deviation of additive noise in temp blocks")
parser.add_argument("--device_ids", type=int, nargs='+', default=[0], help="CUDA device IDs for Fabric")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
args = parser.parse_args()

# ───────────── Path Setup ─────────────
current_dir = args.current_dir
print(current_dir)
arch_dir = os.path.join(current_dir, 'network_arch')
sys.path.append(arch_dir)

from resnet_9 import ResNet9_100_temp_noise

# ───────────── Cutout Augmentation ─────────────
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

# ───────────── Symmetric Quantization ─────────────
def symmetric_quantize(tensor, n_bits=4):
    qmax = 2**(n_bits - 1) - 1
    abs_max = tensor.abs().max()
    scale = abs_max / qmax
    q_tensor = (tensor / scale).round().clamp(-qmax, qmax)
    return q_tensor * scale

# ───────────── Evaluation Function ─────────────
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

# ───────────── Main Function ─────────────
def main():
    fabric = Fabric(accelerator="cuda", devices=[2,3,4,5], strategy="ddp", precision="16-mixed")
    fabric.launch()
    seed_everything(42)

    # Transforms
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    test_transform = tt.Compose([
        tt.ToTensor(),
        tt.Normalize(*stats)
    ])
    test_dataset = CIFAR10(args.data_dir, train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = fabric.setup_dataloaders(test_loader)

    # Instantiate model
    gamma = [16, 50, 35, 35, 60, 70, 80, 200, 50]
    model = ResNet9_100_temp_noise(
        3, out=10, gamma=gamma, std=args.std,
        temp0=True, temp1=True, temp2=True, temp3=True,
        temp4=True, temp5=True, temp6=True, temp7=True, temp8=True
    ).to(fabric.device)

    model.load_state_dict(torch.load(args.checkpoint))

    # Apply quantization
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.data.copy_(symmetric_quantize(param.data, n_bits=args.n_bits))

    model = fabric.setup(model)
    start = time.time()
    print("Starting Evaluation...")
    evaluate(fabric, model, test_loader)
    print(f"Evaluation completed in {(time.time() - start) / 60:.2f} minutes")

if __name__ == "__main__":
    main()
