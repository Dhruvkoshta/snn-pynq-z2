"""
train_direct_encoding.py - Training pipeline with Temporal Input Encoding Optimization

Method 2: Temporal Input Encoding Optimization (Expected Gain: +3% to +5%)
Supports:
  --encoding {direct, poisson}
  --timesteps {4, 8, 16, 32}
"""

import os
import sys
import time
import math
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.snn_direct_encoding import CIFAR10SNN_V4_Direct, simulate_encoding_snr

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import torchvision
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def train_one_epoch(net, dataloader, optimizer, loss_fn, device):
    net.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)

        output_membranes = net(images)  # (T, B, 10)
        final_mem = output_membranes[-1]
        loss = loss_fn(final_mem, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        _, preds = final_mem.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, 100.0 * correct / total


def evaluate_model(net, dataloader, loss_fn, device):
    net.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            output_membranes = net(images)
            final_mem = output_membranes[-1]
            loss = loss_fn(final_mem, targets)

            total_loss += loss.item() * targets.size(0)
            _, preds = final_mem.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

    return total_loss / total, 100.0 * correct / total


def run_training(args):
    print("=" * 70)
    print("  CIFAR-10 SNN: METHOD 2 - TEMPORAL INPUT ENCODING OPTIMIZATION")
    print(f"  Encoding Mode : {args.encoding.upper()}")
    print(f"  Timesteps (T) : {args.timesteps}")
    print(f"  Target Gain   : +3.0% to +5.0% over Poisson baseline")
    print("=" * 70)

    if not TORCH_AVAILABLE:
        print("[!] PyTorch / snnTorch not present in this runtime.")
        print("[+] Performing statistical SNR & temporal variance simulation...")
        metrics = simulate_encoding_snr(np.random.rand(32, 32), timesteps=args.timesteps)
        print(f"[RESULTS] Timesteps T={args.timesteps}:")
        print(f"  Poisson Sampling Variance: {metrics['poisson_var_mean']:.6f}")
        print(f"  Direct Current Variance  : {metrics['direct_var_mean']:.6f}")
        print(f"  Information Fidelity Gain: +{metrics['snr_improvement_db']:.2f} dB")
        print("[*] To train model on GPU, invoke in environment with PyTorch.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device}")

    # Transforms
    cifar_mean = [0.4914, 0.4822, 0.4465]
    cifar_std = [0.2470, 0.2435, 0.2616]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std)
    ])

    train_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    net = CIFAR10SNN_V4_Direct(beta=0.95, encoding=args.encoding, timesteps=args.timesteps).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    def lr_lambda(epoch):
        warmup = args.warmup_epochs
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, (args.epochs - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    save_file = os.path.join(args.save_dir, f"best_snn_{args.encoding}_T{args.timesteps}.pth")

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(net, train_loader, optimizer, loss_fn, device)
        scheduler.step()
        test_loss, test_acc = evaluate_model(net, test_loader, loss_fn, device)
        elapsed = time.time() - t0

        lr_curr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | LR: {lr_curr:.6f} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'state_dict': net.state_dict(), 'acc': best_acc, 'encoding': args.encoding}, save_file)
            print(f"  [+] Saved new checkpoint ({best_acc:.2f}%) -> {save_file}")

    print("\n" + "=" * 70)
    print(f"  TRAINING COMPLETED: Best Test Accuracy = {best_acc:.2f}%")
    print(f"  Poisson Baseline  : ~77.12% (at T=32)")
    print(f"  Observed Gain     : +{best_acc - 77.12:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SNN with Temporal Input Encoding Optimization")
    parser.add_argument("--encoding", type=str, default="direct", choices=["direct", "poisson"],
                        help="Input encoding mechanism: direct (+3-5%%) or baseline poisson")
    parser.add_argument("--timesteps", type=int, default=8, help="Simulation timesteps (direct enables T=4 to 8)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Warmup epochs (default: 3)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--data_dir", type=str, default="./data", help="CIFAR-10 data directory")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Save directory")
    args = parser.parse_args()

    run_training(args)
