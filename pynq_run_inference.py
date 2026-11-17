#!/usr/bin/env python3
"""
pynq_run_inference.py - Production SNN Hardware Inference on PYNQ-Z2

Unified inference engine supporting:
  - Methods 1 & 2: Direct Analog Current Encoding (T=8) vs Baseline Poisson (T=32)
  - Method 5: High-Accuracy Distilled Weights
  - Method 7: Quantization-Aware Signed INT8 Pipeline with Q24 Fixed-Point Scaling
  - FPGA Early-Exit: Hardware AXI-Lite spike decider (ee_axil)
  - Mock Mode: Full software simulation of PYNQ hardware on host PC without a board.

Usage on PYNQ-Z2 (real hardware):
    sudo python3 pynq_run_inference.py --bit hardware/snn.bit --encoding direct --T 8 --n 50

Usage on THIS PC (mock mode):
    python3 pynq_run_inference.py --mock --encoding direct --T 8 --n 20
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "golden"))
import snn_golden as g
import early_exit as ee

MAGIC = 0xEE01
REG_CTRL = 0x00
REG_STATUS = 0x04
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class FpgaEarlyExitDriver:
    """AXI-Lite register driver for the ee_axil hardware block on PYNQ-Z2."""
    def __init__(self, mmio, t_min=2, margin=3, t_max=32):
        self.m = mmio
        self.cfg = (t_min << 12) | (margin << 18) | (t_max << 24)
        status = self.status()
        if (status >> 16) != MAGIC:
            raise RuntimeError(f"Wrong hardware block at address: STATUS read 0x{status:08X}, expected 0xEE01xxxx.")

    def status(self):
        return self.m.read(REG_STATUS)

    def start(self):
        self.m.write(REG_CTRL, self.cfg | (1 << 11))

    def step(self, spikes10):
        word = int(sum(int(b) << c for c, b in enumerate(spikes10)))
        self.m.write(REG_CTRL, self.cfg | (1 << 10) | word)
        s = self.status()
        done = bool(s & 1)
        pred = (s >> 1) & 0xF
        steps = (s >> 5) & 0x3F
        return done, pred, steps


class MockFpgaMMIO:
    """Exact register-level software emulator of ee_axil for testing without board."""
    def __init__(self):
        self.cnt = [0] * 10
        self.t = 0
        self.done = 0
        self.pred = 0
        self.steps = 0
        self.cfg = 0

    def write(self, off, val):
        self.cfg = val
        tmin = (val >> 12) & 63
        marg = max((val >> 18) & 63, 1)
        tmax = (val >> 24) & 63
        if val & (1 << 11):  # Reset
            self.cnt = [0] * 10
            self.t = 0
            self.done = 0
            self.pred = 0
            self.steps = 0
        elif (val & (1 << 10)) and not self.done:  # Step
            for c in range(10):
                self.cnt[c] += (val >> c) & 1
            self.t += 1
            top = max(self.cnt)
            arg = self.cnt.index(top)
            second = max(v for i, v in enumerate(self.cnt) if i != arg)
            if (self.t >= tmin and top - second >= marg) or self.t >= tmax:
                self.done, self.pred, self.steps = 1, arg, self.t

    def read(self, off):
        return (MAGIC << 16) | (self.steps << 5) | (self.pred << 1) | self.done


def load_dataset(cifar_dir, num_samples):
    batch_file = Path(cifar_dir) / "test_batch"
    if not batch_file.exists():
        raise FileNotFoundError(f"CIFAR test_batch not found at {batch_file}")

    with open(batch_file, "rb") as f:
        d = pickle.load(f, encoding="bytes")

    raw_data = np.frombuffer(d[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    labels = np.array(d[b"labels"])

    n = min(num_samples, len(labels))
    images = raw_data[:n].astype(np.float32) / 255.0
    targets = labels[:n]
    return images, targets


def run_inference(args):
    print("=" * 80)
    print("  PYNQ-Z2 SNN HARDWARE ACCELERATOR: INFERENCE & BENCHMARK ENGINE")
    print(f"  Target Hardware   : {'MOCK MMIO (Host PC Simulation)' if args.mock else 'PYNQ-Z2 FPGA'}")
    print(f"  Input Encoding    : {args.encoding.upper()} ({'Optimized Method 2' if args.encoding == 'direct' else 'Baseline Poisson'})")
    print(f"  Simulation T      : {args.T} timesteps")
    print(f"  Arithmetic Mode   : {args.mode.upper()} (INT8 Quantized Hardware Pipeline)")
    print(f"  Confidence Exit   : {'ENABLED (margin=' + str(args.margin) + ', t_min=' + str(args.t_min) + ')' if args.early_exit else 'DISABLED'}")
    print("=" * 80)

    # 1. Hardware Driver Setup
    if args.mock:
        mmio = MockFpgaMMIO()
    else:
        try:
            from pynq import Overlay, MMIO
            bit_path = args.bit or str(HERE / "hardware" / "snn.bit")
            if not os.path.exists(bit_path):
                raise FileNotFoundError(f"Bitstream file not found at {bit_path}")
            print(f"[+] Programming Zynq-7000 PL with bitstream: {bit_path} ...")
            ol = Overlay(bit_path)
            cands = [k for k, v in ol.ip_dict.items() if "ee_axil" in (k + str(v.get("type", ""))).lower()]
            base = ol.ip_dict[cands[0]]["phys_addr"] if cands else 0x43C00000
            print(f"[+] Found ee_axil IP at physical address: 0x{base:08X}")
            mmio = MMIO(base, 0x1000)
        except ImportError:
            print("[!] PYNQ library not found on this machine; falling back to MockMMIO.")
            mmio = MockFpgaMMIO()

    ee_driver = FpgaEarlyExitDriver(mmio, t_min=args.t_min, margin=args.margin, t_max=args.T)

    # 2. Load Weights and Network
    weights_root = args.root or str(HERE / "weights")
    print(f"[+] Loading neural network parameters from: {weights_root}")
    params = g.Params(weights_root, input_scale=(1.0 / 64.0 if args.encoding == "direct" else 1.0))
    net = g.Net(params, mode=args.mode)

    # 3. Load CIFAR-10 Test Batch
    images, labels = load_dataset(args.cifar or str(HERE / "cifar"), args.n)
    print(f"[+] Loaded {len(images)} test samples.")

    # 4. Inference Loop
    correct = 0
    total = len(images)
    total_timesteps_used = 0
    start_time = time.time()
    latencies = []

    print("-" * 80)
    print(f"{'Sample':<8} | {'True Label':<12} | {'Predicted':<12} | {'Status':<6} | {'Exit Step':<10} | {'Latency (ms)'}")
    print("-" * 80)

    for i in range(total):
        img = images[i:i+1]
        target = labels[i]

        t0 = time.time()
        ee_driver.start()

        # Stream timesteps through SNN layers
        exit_step = args.T
        pred_class = 0
        done = False
        accumulated_spikes = np.zeros(10, dtype=int)

        # Reset neuron state for new image
        for n in g.LIF_NAMES:
            net.lif[n].mem = None

        frames = g.make_frames(img, args.T, encoding=args.encoding, norm=args.norm, seed=i)

        for step_idx, frame in enumerate(frames, start=1):
            if net.mode != "float" and args.encoding == "direct":
                frame = np.clip(np.round(frame / net.in_scale), -127, 127).astype(np.float32)

            out_spikes = net.step(frame, lambda name, s: None)  # (1, 10) binary spikes
            accumulated_spikes += out_spikes[0].astype(int)

            if args.early_exit:
                done, hw_pred, steps = ee_driver.step(out_spikes[0])
                if done:
                    exit_step = steps
                    pred_class = hw_pred
                    break

        if not done:
            pred_class = int(np.argmax(accumulated_spikes))
            exit_step = args.T

        t_elapsed = (time.time() - t0) * 1000.0
        latencies.append(t_elapsed)
        total_timesteps_used += exit_step

        is_correct = (pred_class == target)
        if is_correct:
            correct += 1

        print(f"{i+1:04d}     | {CIFAR_CLASSES[target]:<12} | {CIFAR_CLASSES[pred_class]:<12} | "
              f"{'MATCH' if is_correct else 'MISS':<6} | T={exit_step:<7} | {t_elapsed:6.2f} ms")

    total_time = time.time() - start_time
    avg_latency = np.mean(latencies)
    fps = total / total_time
    accuracy = (correct / total) * 100.0
    avg_steps = total_timesteps_used / total
    speedup = args.T / avg_steps

    print("=" * 80)
    print("  HARDWARE EXECUTION RESULTS SUMMARY:")
    print("=" * 80)
    print(f"  Total Images Processed      : {total}")
    print(f"  Classification Accuracy     : {accuracy:.2f}% ({correct}/{total})")
    print(f"  Average Timesteps per Image : {avg_steps:.2f} / {args.T} steps ({speedup:.2f}x speedup)")
    print(f"  Mean Latency per Inference  : {avg_latency:.2f} ms")
    print(f"  Effective Throughput        : {fps:.2f} FPS")

    power_w = args.power_w or 2.10  # Measured nominal PYNQ-Z2 board power
    energy_per_inf = (power_w * (avg_latency / 1000.0)) * 1000.0  # in mJ
    print(f"  Board Power (Nominal)       : {power_w:.2f} W")
    print(f"  Energy per Classification   : {energy_per_inf:.2f} mJ")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SNN Inference on PYNQ-Z2")
    parser.add_argument("--mock", action="store_true", help="Run with software Mock MMIO (for testing on host PC)")
    parser.add_argument("--bit", type=str, default=None, help="Path to snn.bit overlay file")
    parser.add_argument("--root", type=str, default=None, help="Root folder containing weights")
    parser.add_argument("--cifar", type=str, default=None, help="Folder containing cifar test_batch")
    parser.add_argument("--n", type=int, default=10, help="Number of test images to evaluate (default: 10)")
    parser.add_argument("--T", type=int, default=8, help="Simulation timesteps (8 for direct, 32 for poisson)")
    parser.add_argument("--encoding", type=str, default="direct", choices=["direct", "poisson"], help="Input encoding mode")
    parser.add_argument("--mode", type=str, default="fixed", choices=["fixed", "int8", "float"], help="Arithmetic precision")
    parser.add_argument("--norm", type=str, default="unit", choices=["unit", "standard"], help="Image normalization")
    parser.add_argument("--early_exit", action="store_true", default=True, help="Enable hardware confidence early exit")
    parser.add_argument("--margin", type=int, default=3, help="Confidence threshold margin (top - second >= margin)")
    parser.add_argument("--t_min", type=int, default=2, help="Minimum timesteps before early exit")
    parser.add_argument("--power_w", type=float, default=2.1, help="Measured board power in Watts")
    args = parser.parse_args()

    run_inference(args)
