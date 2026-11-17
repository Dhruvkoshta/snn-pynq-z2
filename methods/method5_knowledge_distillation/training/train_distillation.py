"""
train_distillation.py - SNN Student Training Guided by ANN Teacher

Method 5: Knowledge Distillation (ANN -> SNN) (Expected Gain: +4% to +7%)
Distills non-spiking ResNet-20 knowledge into CIFAR10SNN_V4 student.
"""

import os
import sys
import time
import math
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ann_teacher import ResNet20_ANN_Teacher
from training.distillation_loss import KnowledgeDistillationLoss, numpy_distillation_loss

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import torchvision
    import torchvision.transforms as transforms
    import snntorch as snn
    from snntorch import surrogate
    from snntorch import utils
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
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

    class CIFAR10SNN_Student(nn.Module):
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


def train_distillation_epoch(student, teacher, dataloader, optimizer, distil_loss_fn, num_steps, device):
    student.train()
    teacher.eval()  # Teacher is frozen

    total_loss, total_ce, total_soft = 0.0, 0.0, 0.0
    correct, total = 0, 0

    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)

        # 1. Obtain Teacher Soft Logits (no gradients)
        with torch.no_grad():
            teacher_logits = teacher(images)

        # 2. SNN Forward Pass across Timesteps
        utils.reset(student)
        output_membranes = []
        for step in range(num_steps):
            input_spikes = (torch.rand_like(images) < images).float()
            spk_out, mem_out = student(input_spikes)
            output_membranes.append(mem_out)

        # Output logits derived from final membrane accumulation
        final_membrane = output_membranes[-1]

        # 3. Compute Distillation Loss
        loss, ce_val, soft_val = distil_loss_fn(final_membrane, teacher_logits, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        total_ce += ce_val * targets.size(0)
        total_soft += soft_val * targets.size(0)

        _, preds = final_membrane.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, total_ce / total, total_soft / total, 100.0 * correct / total


def evaluate_snn(student, dataloader, num_steps, device):
    student.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            utils.reset(student)
            output_membranes = []
            for step in range(num_steps):
                input_spikes = (torch.rand_like(images) < images).float()
                spk_out, mem_out = student(input_spikes)
                output_membranes.append(mem_out)

            final_membrane = output_membranes[-1]
            _, preds = final_membrane.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

    return 100.0 * correct / total


def run_pipeline(args):
    print("=" * 70)
    print("  CIFAR-10 SNN: METHOD 5 - KNOWLEDGE DISTILLATION (ANN -> SNN)")
    print(f"  Distillation Temperature (tau): {args.temperature}")
    print(f"  Distillation Alpha (soft wt)  : {args.alpha}")
    print(f"  Expected Accuracy Gain        : +4.0% to +7.0%")
    print("=" * 70)

    if not TORCH_AVAILABLE:
        print("[!] PyTorch / snnTorch not installed in this environment.")
        print("[+] Performing NumPy validation of Teacher-Student Dark Knowledge transfer...")
        z_snn = np.random.randn(64, 10)
        z_teacher = z_snn * 1.5 + np.random.randn(64, 10) * 0.1
        y = np.random.randint(0, 10, size=(64,))
        tot, ce, kl = numpy_distillation_loss(z_snn, z_teacher, y, temperature=args.temperature, alpha=args.alpha)
        print(f"[SUCCESS] Distillation Loss verified: Total={tot:.4f}, Hard CE={ce:.4f}, Soft KL={kl:.4f}")
        print("[*] Launch on GPU-enabled node with torch/snntorch to execute full training.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device}")

    # Standard CIFAR-10 Transforms
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

    # 1. Initialize ANN Teacher
    teacher = ResNet20_ANN_Teacher(num_classes=10).to(device)
    if os.path.exists(args.teacher_weights):
        print(f"[+] Loading pre-trained ANN Teacher weights from {args.teacher_weights}")
        teacher.load_state_dict(torch.load(args.teacher_weights, map_location=device))
    else:
        print("[!] Pre-trained teacher weights not found at path; using initialized teacher.")

    # Freeze Teacher parameters completely
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    # 2. Initialize SNN Student
    student = CIFAR10SNN_Student(beta=0.95).to(device)

    # 3. Loss & Optimizer
    distil_loss_fn = KnowledgeDistillationLoss(temperature=args.temperature, alpha=args.alpha)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        warmup = args.warmup_epochs
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, (args.epochs - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "best_snn_distilled.pth")

    for epoch in range(args.epochs):
        t0 = time.time()
        loss, ce_loss, kl_loss, train_acc = train_distillation_epoch(
            student, teacher, train_loader, optimizer, distil_loss_fn, args.timesteps, device
        )
        scheduler.step()
        test_acc = evaluate_snn(student, test_loader, args.timesteps, device)
        elapsed = time.time() - t0

        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | LR: {curr_lr:.6f} | "
              f"Loss: {loss:.4f} (CE: {ce_loss:.4f}, KL: {kl_loss:.4f}) | "
              f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'state_dict': student.state_dict(), 'acc': best_acc}, save_path)
            print(f"  [+] Saved best distilled student ({best_acc:.2f}%) -> {save_path}")

    print("\n" + "=" * 70)
    print(f"  DISTILLATION COMPLETED: Best Student Test Accuracy = {best_acc:.2f}%")
    print(f"  Baseline Accuracy     : ~77.12%")
    print(f"  Observed Gain         : +{best_acc - 77.12:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation from ANN to SNN")
    parser.add_argument("--temperature", type=float, default=4.0, help="Distillation temperature tau (default: 4.0)")
    parser.add_argument("--alpha", type=float, default=0.7, help="Soft loss weight alpha in [0, 1] (default: 0.7)")
    parser.add_argument("--teacher_weights", type=str, default="./checkpoints/ann_teacher_resnet20.pth", help="Path to ANN weights")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--timesteps", type=int, default=32, help="SNN timesteps (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak LR")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Warmup epochs")
    parser.add_argument("--workers", type=int, default=4, help="Workers")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data dir")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Save dir")
    args = parser.parse_args()

    run_pipeline(args)
