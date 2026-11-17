"""
qat_snn.py - CIFAR-10 SNN Architecture with Quantization-Aware Training (QAT)

Integrates QATConv2d and QATLinear layers into CIFAR10SNN_V4 to support Straight-Through
Estimator (STE) training directly targeting PYNQ-Z2 hardware registers.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.qat_modules import QATConv2d, QATLinear

try:
    import torch
    import torch.nn as nn
    import snntorch as snn
    from snntorch import surrogate
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class QATResidualSNNBlock(nn.Module):
        def __init__(self, channels, beta=0.95, spike_grad=None, num_bits=8):
            super().__init__()
            if spike_grad is None:
                spike_grad = surrogate.fast_sigmoid(slope=25)

            self.conv1 = QATConv2d(channels, channels, kernel_size=3, padding=1, bias=False, num_bits=num_bits)
            self.bn1 = nn.BatchNorm2d(channels)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)

            self.conv2 = QATConv2d(channels, channels, kernel_size=3, padding=1, bias=False, num_bits=num_bits)
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


    class CIFAR10SNN_V4_QAT(nn.Module):
        def __init__(self, beta=0.95, num_bits=8):
            super().__init__()
            grad = surrogate.fast_sigmoid(slope=25)

            self.conv0 = QATConv2d(3, 64, kernel_size=3, padding=1, bias=False, num_bits=num_bits)
            self.bn0 = nn.BatchNorm2d(64)
            self.lif0 = snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True)
            self.pool0 = nn.MaxPool2d(2)

            self.res1 = QATResidualSNNBlock(64, beta, grad, num_bits=num_bits)

            self.conv1 = QATConv2d(64, 128, kernel_size=3, padding=1, bias=False, num_bits=num_bits)
            self.bn1 = nn.BatchNorm2d(128)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True)
            self.pool1 = nn.MaxPool2d(2)

            self.res2 = QATResidualSNNBlock(128, beta, grad, num_bits=num_bits)

            self.conv2 = QATConv2d(128, 256, kernel_size=3, padding=1, bias=False, num_bits=num_bits)
            self.bn2 = nn.BatchNorm2d(256)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True)
            self.pool2 = nn.MaxPool2d(2)

            self.res3 = QATResidualSNNBlock(256, beta, grad, num_bits=num_bits)

            self.fc = QATLinear(256 * 4 * 4, 10, num_bits=num_bits)
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

        def set_quantization(self, enabled=True):
            for m in self.modules():
                if isinstance(m, (QATConv2d, QATLinear)):
                    m.quantize_enabled = enabled
else:
    class CIFAR10SNN_V4_QAT:
        def __init__(self, *args, **kwargs):
            pass


if __name__ == "__main__":
    if TORCH_AVAILABLE:
        model = CIFAR10SNN_V4_QAT()
        print("[+] CIFAR10SNN_V4_QAT initialized successfully with Straight-Through Estimators.")
    else:
        print("[*] PyTorch not loaded; verified module class registration.")
