# AGENT_GUIDE.md - AI Agent Operating Manual & Repository Specification

Welcome to the **PYNQ-Z2 SNN Hardware Accelerator** repository. This document serves as the ground-truth technical guide for autonomous AI agents and developers interacting with, modifying, or extending this codebase.

---

## 1. System Architecture & Context

This repository hosts a production **Spiking Neural Network (SNN)** accelerator designed for CIFAR-10 image classification deployed on the **Xilinx PYNQ-Z2 (Zynq-7000 SoC)**.

### Hardware-Software Co-Design:
* **Processing System (ARM Cortex-A9)**: Runs the PyTorch / NumPy SNN layers (convolution, batch norm, leaky integrate-and-fire membrane dynamics).
* **Programmable Logic (FPGA PL)**: Implements the `ee_axil.v` AXI4-Lite hardware accelerator that monitors output class spikes in real time and computes the **Confidence-Gated Early Exit** decision $(S_{top} - S_{second} \ge \text{margin})$.
* **Communication**: AXI-Lite MMIO register space mapped at base address `0x43C00000` (or dynamically discovered via `pynq.Overlay.ip_dict`).

---

## 2. Directory & Component Layout

```
.
├── hardware/                           # FPGA Hardware Handoff
│   ├── snn.bit                         # Synthesized Bitstream for Zynq-7000 XC7Z020
│   └── snn.hwh                         # Vivado Hardware Handoff definition
├── weights/                            # Neural Network Parameters
│   ├── weights_float32/                # FP32 weights, bias, and BatchNorm parameters
│   ├── weights_int8_experiment/        # Signed INT8 [-128, 127] weights + manifest.json scales
│   ├── weights_all_combined_int8.bin   # QAT-trained 1.87 MB deployment binary table
│   └── snn_combined_config.h           # Q24 scale integers & register offsets
├── cifar/                              # Evaluation Dataset
│   └── test_batch                      # Python pickle containing 500 CIFAR-10 test samples
├── golden/                             # Bit-Accurate Reference Simulator
│   ├── snn_golden.py                   # Fixed-point Q16 / Float32 bit-accurate SNN model
│   └── early_exit.py                   # Reference software early exit decider
├── methods/
│   └── all_combined/                   # Master unified model & training pipeline (All 4 methods)
│       ├── models/                     # SNN combined architecture & ANN Teacher
│       ├── training/                   # Data augmentation, distillation loss & QAT STE
│       ├── validation/                 # Quantitative ablation benchmark
│       └── export/                     # PYNQ-Z2 hardware exporter
├── run_all_methods_overall.py          # Master evaluation runner (90% accuracy, 4x speedup)
├── pynq_run_inference.py               # Production hardware inference runner
├── test_on_this_machine.py             # Pre-flight self-test suite (runs without board)
├── requirements.txt                    # Python dependencies
├── README.md                           # Human-readable guide & benchmark tables
└── AGENT_GUIDE.md                      # This file
```

---

## 3. FPGA Register Map (`ee_axil`)

When modifying the hardware interface in `pynq_run_inference.py`, adhere strictly to the register map defined in `rtl/ee_axil.v`:

| Offset | Register Name | R/W | Bit Field Layout & Function |
| :---: | :---: | :---: | :--- |
| `0x00` | `REG_CTRL` | W | `[9:0]`: 10-bit spike word (`spikes10`)<br>`[10]`: Step strobe (`1` = clock tick)<br>`[11]`: Reset strobe (`1` = clear counters)<br>`[17:12]`: `t_min` (6 bits, default: 2)<br>`[23:18]`: `margin` (6 bits, default: 3)<br>`[29:24]`: `t_max` (6 bits, default: 32) |
| `0x04` | `REG_STATUS` | R | `[0]`: `done` flag (`1` = early exit reached or $t=T$)<br>`[4:1]`: `predicted_class` (4 bits, 0-9)<br>`[10:5]`: `steps_taken` (6 bits, 1-32)<br>`[31:16]`: `MAGIC` signature (Always `0xEE01`) |

### Hardware Verification Invariant:
Before writing to `REG_CTRL`, verify that `(REG_STATUS >> 16) == 0xEE01`. If not, the FPGA overlay is unprogrammed or the base address is incorrect.

---

## 4. Dual-Mode Execution Workflow for Agents

An AI agent working on this repository may be running on the **developer's host PC** (where `pynq` is not installed) or **directly on the PYNQ-Z2 board**. The code is designed to support both seamlessly:

### Mode A: Host PC Simulation (Mock Mode)
* Uses `MockFpgaMMIO` to emulate the exact register transitions of `ee_axil.v` in software.
* Command:
  ```bash
  python3 pynq_run_inference.py --mock --encoding direct --T 8 --n 10
  ```
* Automated Pre-flight test:
  ```bash
  python3 test_on_this_machine.py
  ```

### Mode B: Physical PYNQ-Z2 FPGA Execution
* Automatically loads `pynq.Overlay("hardware/snn.bit")` and accesses physical registers at `0x43C00000`.
* Command:
  ```bash
  sudo python3 pynq_run_inference.py --bit hardware/snn.bit --encoding direct --T 8 --n 50 --early_exit
  ```

---

## 5. How to Train, Evaluate & Export Changes

### 1. Evaluating the Combined Accuracy Ablation Matrix:
```bash
python3 methods/all_combined/software/validation/evaluate_ablation.py
```

### 2. Retraining with All 4 Improvements:
```bash
python3 methods/all_combined/software/training/train_all_combined.py \
    --use_aug --aug_policy advanced \
    --encoding direct --timesteps 8 \
    --use_distill --temperature 4.0 --alpha 0.7 \
    --use_qat --epochs 50 --batch_size 64
```

### 3. Re-exporting FPGA Binary Weights:
If model weights are retrained, re-generate the FPGA binary tables and C headers:
```bash
python3 methods/all_combined/software/export/export_all_to_fpga.py
```
This generates:
* `weights/weights_all_combined_int8.bin` (Raw INT8 weights array, 1.87 MB)
* `weights/snn_combined_config.h` (Q24 fixed-point scaling multipliers)

---

## 6. Guidelines for Autonomous Agents

1. **Do not break mock compatibility**: When editing `pynq_run_inference.py` or `golden/snn_golden.py`, ensure `test_on_this_machine.py` still exits with code `0`.
2. **Preserve INT8 Alignment**: The FPGA MAC expects signed 8-bit weights in range $[-128, 127]$. Do not alter tensor quantization to unsigned uint8 without updating RTL.
3. **Keep Timesteps $T \le 32$**: The hardware counter register field `[10:5]` is 6 bits wide (supports up to $T=63$). Keep $T \in [4, 32]$.
4. **Clean Git Hygiene**: Do not commit `.pyc`, `__pycache__`, or `.zip` files into git.
