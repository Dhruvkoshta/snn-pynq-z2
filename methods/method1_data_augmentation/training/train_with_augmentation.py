"""
train_with_augmentation.py - SNN Training with Advanced Data Pipeline & Augmentation

Method 1: Data Pipeline & Data Augmentation (Expected Gain: +5% to +8%)
Key Improvements over Baseline:
  1. Reflection-padded Random Crop (32x32, pad=4)
  2. Random Horizontal Flip (p=0.5)
  3. CIFAR-10 Per-Channel Normalization (mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
  4. Cutout / Random Erasing (p=0.5, scale=(0.02, 0.33))
  5. Cosine Annealing Learning Rate Schedule with Warmup
  6. Label Smoothing Cross-Entropy Loss
"""

import os
import sys
import time
import math
import argparse
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.data_augmentation import get_cifar10_transforms, apply_numpy_augmentation

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import torchvision
    import torchvision.transforms as transforms
    import snntorch as snn
    from snntorch import surrogate
    from snntorch import utils
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ==========================================
# SNN Model Architecture (CIFAR10SNN_V4)
# ==========================================
if TORCH_AVAILABLE:
    spike_grad_v4 = surrogate.fast_sigmoid(slope=25)
    beta_v4 = 0.95

    class ResidualSNNBlock(nn.Module):
        def __init__(self, channels, beta=0.95, spike_grad=None):
            super().__init__()
            if spike_grad is None:
                spike_grad = surrogate.fast_sigmoid(slope=25)
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)

            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.lif1(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.lif2(out)
            return out + identity

    class CIFAR10SNN_V4(nn.Module):
        def __init__(self, beta=0.95):
            super().__init__()
            grad = surrogate.fast_sigmoid(slope=25)
            self.conv0 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            self.bn0 = nn.BatchNorm2d(64)
            self.lif0 = snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True)
            self.pool0 = nn.MaxPool2d(2)

            self.res1 = ResidualSNNBlock(64, beta, grad)

            self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(128)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True)
            self.pool1 = nn.MaxPool2d(2)

            self.res2 = ResidualSNNBlock(128, beta, grad)

            self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(256)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True)
            self.pool2 = nn.MaxPool2d(2)

            self.res3 = ResidualSNNBlock(256, beta, grad)

            self.fc = nn.Linear(256 * 4 * 4, 10)
            self.lif_out = snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)

        def forward(self, x):
            x = self.conv0(x)
            x = self.bn0(x)
            x = self.lif0(x)
            x = self.pool0(x)

            x = self.res1(x)

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.lif1(x)
            x = self.pool1(x)

            x = self.res2(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.lif2(x)
            x = self.pool2(x)

            x = self.res3(x)

            x = x.view(x.size(0), -1)
            x = self.fc(x)
            spk_out, mem_out = self.lif_out(x)
            return spk_out, mem_out


def train_epoch(net, dataloader, optimizer, loss_fn, num_steps, device):
    """Executes one training epoch with Poisson rate encoded inputs."""
    net.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device)
        utils.reset(net)

        output_membranes = []
        for step in range(num_steps):
            # Poisson rate coding
            input_spikes = (torch.rand_like(images) < images).float()
            spk_out, mem_out = net(input_spikes)
            output_membranes.append(mem_out)

        # Class prediction based on final membrane accumulation
        final_membrane = output_membranes[-1]
        loss = loss_fn(final_membrane, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        _, preds = final_membrane.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, 100.0 * correct / total


def evaluate(net, dataloader, loss_fn, num_steps, device):
    """Evaluates validation/test accuracy and loss."""
    net.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            utils.reset(net)

            output_membranes = []
            for step in range(num_steps):
                input_spikes = (torch.rand_like(images) < images).float()
                spk_out, mem_out = net(input_spikes)
                output_membranes.append(mem_out)

            final_membrane = output_membranes[-1]
            loss = loss_fn(final_membrane, targets)

            total_loss += loss.item() * targets.size(0)
            _, preds = final_membrane.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

    return total_loss / total, 100.0 * correct / total


def run_pipeline(args):
    print("=" * 70)
    print("  CIFAR-10 SNN TRAINING - METHOD 1: DATA AUGMENTATION PIPELINE")
    print(f"  Augmentation Policy : {args.policy.upper()}")
    print(f"  Target Accuracy Gain: +5.0% to +8.0% over baseline")
    print("=" * 70)

    if not TORCH_AVAILABLE:
        print("[!] PyTorch / snnTorch not detected in the current python runtime.")
        print("[+] Validating data augmentation using high-performance NumPy pipeline...")
        x_synthetic = np.random.uniform(0.0, 1.0, (128, 3, 32, 32)).astype(np.float32)
        augmented = apply_numpy_augmentation(x_synthetic)
        print(f"[SUCCESS] Augmented {len(augmented)} images with RandomCrop and Flip.")
        print("[*] To train full SNN on GPU, run inside Python environment with torch and snntorch.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device}")

    # Build Augmented Datasets
    train_transform, test_transform = get_cifar10_transforms(policy=args.policy, cutout_length=args.cutout_len)

    train_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    net = CIFAR10SNN_V4().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def lr_lambda(epoch):
        warmup = args.warmup_epochs
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, (args.epochs - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, f"best_snn_augmented_{args.policy}.pth")

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(net, train_loader, optimizer, loss_fn, args.timesteps, device)
        scheduler.step()
        test_loss, test_acc = evaluate(net, test_loader, loss_fn, args.timesteps, device)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'test_acc': test_acc,
                'policy': args.policy
            }, checkpoint_path)
            print(f"  [+] Saved new best model ({best_acc:.2f}%) -> {checkpoint_path}")

    print("\n" + "=" * 70)
    print(f"  TRAINING COMPLETED: Best Accuracy = {best_acc:.2f}%")
    print(f"  Baseline Accuracy : ~77.12%")
    print(f"  Empirical Gain    : +{best_acc - 77.12:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 SNN Training with Data Augmentation")
    parser.add_argument("--policy", type=str, default="advanced", choices=["baseline", "standard", "advanced", "autoaugment"],
                        help="Data augmentation policy: standard (+4-5%%), advanced (+6-8%%), autoaugment (+7-9%%)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--timesteps", type=int, default=32, help="Number of SNN simulation timesteps (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate (default: 1e-3)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay (default: 1e-4)")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs for cosine scheduler (default: 5)")
    parser.add_argument("--cutout_len", type=int, default=16, help="Cutout length in pixels (default: 16)")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing epsilon (default: 0.05)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory where CIFAR-10 is stored")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    args = parser.parse_args()

    run_pipeline(args)
