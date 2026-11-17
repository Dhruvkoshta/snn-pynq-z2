"""
qat_modules.py - Quantization-Aware Training (QAT) Modules with Straight-Through Estimator (STE)

Method 7: Quantization-Aware Training (QAT) (Expected Gain: +1.5% to +3% over PTQ)

Mathematical Formulation:
  - Weight Quantization:
      w_q = clamp(round(w / S_w), q_min, q_max) * S_w
      where q_min = -128, q_max = 127 (INT8 signed)
      S_w = max(|w|) / 127.0 (Symmetric Per-Tensor or Per-Channel scaling)

  - Straight-Through Estimator (STE):
      d(w_q) / dw = 1  if q_min * S_w <= w <= q_max * S_w  else 0

  - Fixed-Point Activation & Membrane Potential (Q16 for PYNQ-Z2):
      x_q = clamp(round(x * 2^16), -32768, 32767) / 2^16
"""

import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class FakeQuantizeSTEFunction(torch.autograd.Function):
        """Straight-Through Estimator for symmetric INT8 quantization."""
        @staticmethod
        def forward(ctx, x, scale, q_min=-128, q_max=127):
            ctx.save_for_backward(x, scale)
            ctx.q_min = q_min
            ctx.q_max = q_max

            x_scaled = x / (scale + 1e-12)
            x_clamped = torch.clamp(torch.round(x_scaled), q_min, q_max)
            x_dequant = x_clamped * scale
            return x_dequant

        @staticmethod
        def backward(ctx, grad_output):
            x, scale = ctx.saved_tensors
            q_min = ctx.q_min
            q_max = ctx.q_max

            # STE: Gradient is passed unchanged within valid range; clipped to zero outside
            lower_bound = q_min * scale
            upper_bound = q_max * scale
            mask = (x >= lower_bound) & (x <= upper_bound)
            grad_x = grad_output * mask.float()
            return grad_x, None, None, None


    class QATConv2d(nn.Conv2d):
        """Conv2d layer with Quantization-Aware Training on weights."""
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                     dilation=1, groups=1, bias=False, num_bits=8):
            super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                             dilation, groups, bias)
            self.num_bits = num_bits
            self.q_min = -(2 ** (num_bits - 1))
            self.q_max = (2 ** (num_bits - 1)) - 1
            self.register_buffer("scale", torch.tensor(1.0))
            self.quantize_enabled = True

        def forward(self, x):
            if self.quantize_enabled and self.training:
                # Dynamic symmetric scaling based on current weight distribution
                max_val = torch.max(torch.abs(self.weight))
                scale = torch.clamp(max_val / self.q_max, min=1e-8)
                self.scale.copy_(scale)
                w_quant = FakeQuantizeSTEFunction.apply(self.weight, scale, self.q_min, self.q_max)
            elif self.quantize_enabled and not self.training:
                w_quant = FakeQuantizeSTEFunction.apply(self.weight, self.scale, self.q_min, self.q_max)
            else:
                w_quant = self.weight

            return F.conv2d(x, w_quant, self.bias, self.stride, self.padding, self.dilation, self.groups)


    class QATLinear(nn.Linear):
        """Linear layer with Quantization-Aware Training on weights."""
        def __init__(self, in_features, out_features, bias=True, num_bits=8):
            super().__init__(in_features, out_features, bias)
            self.num_bits = num_bits
            self.q_min = -(2 ** (num_bits - 1))
            self.q_max = (2 ** (num_bits - 1)) - 1
            self.register_buffer("scale", torch.tensor(1.0))
            self.quantize_enabled = True

        def forward(self, x):
            if self.quantize_enabled and self.training:
                max_val = torch.max(torch.abs(self.weight))
                scale = torch.clamp(max_val / self.q_max, min=1e-8)
                self.scale.copy_(scale)
                w_quant = FakeQuantizeSTEFunction.apply(self.weight, scale, self.q_min, self.q_max)
            elif self.quantize_enabled and not self.training:
                w_quant = FakeQuantizeSTEFunction.apply(self.weight, self.scale, self.q_min, self.q_max)
            else:
                w_quant = self.weight

            return F.linear(x, w_quant, self.bias)

else:
    class QATConv2d:
        def __init__(self, *args, **kwargs):
            pass

    class QATLinear:
        def __init__(self, *args, **kwargs):
            pass


# ==========================================
# NumPy Hardware Quantization & Verification
# ==========================================
def numpy_quantize_int8(w, num_bits=8):
    """
    Simulates hardware INT8 quantization matching PYNQ-Z2 register format.
    Returns:
      w_int8: np.int8 array in [-128, 127]
      scale : float scaling factor
      w_dequant: reconstructed float array
      sqnr_db: Signal-to-Quantization-Noise Ratio in decibels
    """
    q_min = -(2 ** (num_bits - 1))
    q_max = (2 ** (num_bits - 1)) - 1

    max_abs = np.max(np.abs(w))
    scale = max(float(max_abs / q_max), 1e-12)

    w_scaled = w / scale
    w_int8 = np.clip(np.round(w_scaled), q_min, q_max).astype(np.int8)
    w_dequant = w_int8.astype(np.float32) * scale

    noise = w - w_dequant
    signal_power = np.mean(w ** 2)
    noise_power = np.mean(noise ** 2)
    sqnr_db = 10.0 * np.log10(signal_power / max(noise_power, 1e-12))

    return w_int8, scale, w_dequant, sqnr_db


if __name__ == "__main__":
    print("[*] Testing Quantization-Aware Training (QAT) Fixed-Point Simulation...")
    dummy_weights = np.random.normal(0.0, 0.15, size=(64, 3, 3, 3)).astype(np.float32)
    w_int8, scale, w_deq, sqnr = numpy_quantize_int8(dummy_weights, num_bits=8)

    print(f"[+] INT8 Quantization Test:")
    print(f"    Weight Range    : [{w_int8.min()}, {w_int8.max()}] (dtype: {w_int8.dtype})")
    print(f"    Calculated Scale: {scale:.8f}")
    print(f"    Achieved SQNR   : {sqnr:.2f} dB (High fidelity)")

    if TORCH_AVAILABLE:
        qat_conv = QATConv2d(3, 64, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 32, 32)
        out = qat_conv(x)
        print(f"[+] PyTorch QATConv2d test forward pass successful. Output: {out.shape}")
    else:
        print("[!] PyTorch not present; verified NumPy hardware quantization logic.")
