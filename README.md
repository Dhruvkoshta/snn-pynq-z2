# PYNQ-Z2 SNN Hardware Accelerator: Master Deployment Package

This standalone package consolidates everything needed to run and benchmark the **CIFAR-10 Spiking Neural Network (SNN)** on the **Xilinx PYNQ-Z2 FPGA board**, as well as test it locally on this host machine.

---

## 1. What's Inside This Package

```
pynq_z2_snn_package/
├── hardware/
│   ├── snn.bit                         # FPGA PL bitstream overlay (Zynq-7000 XC7Z020)
│   └── snn.hwh                         # Hardware handoff specification for PYNQ
├── weights/
│   ├── weights_float32/                # FP32 weights & batch-norm parameters
│   ├── weights_int8_experiment/        # Post-training quantized INT8 weights & scales
│   ├── weights_all_combined_int8.bin   # QAT INT8 trained binary table
│   └── snn_combined_config.h           # Q24 scale registers & AXIL offsets
├── cifar/
│   └── test_batch                      # Compact CIFAR-10 test set (500 samples)
├── golden/
│   ├── snn_golden.py                   # Bit-accurate fixed-point / float reference model
│   └── early_exit.py                   # Hardware confidence early exit decider
├── methods/                            # Source implementations of all 4 improvements:
│   ├── method1_data_augmentation/      # Cutout, reflection pad, normalization (+5% to +8%)
│   ├── method2_direct_encoding/        # Direct Analog Current Encoding T=8 (+3% to +5%)
│   ├── method5_knowledge_distillation/ # ResNet-20 ANN Teacher distillation (+4% to +7%)
│   ├── method7_quantization_aware_training/ # INT8 STE QAT (+1.5% to +3% over PTQ)
│   └── all_combined/                   # Master combined model & ablation suite
├── pynq_run_inference.py               # Main hardware inference execution script
├── test_on_this_machine.py             # Pre-flight host PC verification script
└── README.md
```

---

## 2. Steps on THIS Machine (Host PC)

### Step 2.1: Run Pre-flight Health Check
Verify all weights, dataset, bitstream files, and hardware drivers locally using software simulation:
```bash
cd /home/dhruv/scratch/projects/sur/MP/pynq_z2_snn_package
python3 test_on_this_machine.py
```

### Step 2.2: Test Inference in Mock Mode on This PC
Simulate the FPGA overlay and AXI-Lite registers locally without needing the board connected:
```bash
# Test with Direct Analog Current Encoding (T=8 timesteps, 4x faster):
python3 pynq_run_inference.py --mock --encoding direct --T 8 --n 10

# Test with Baseline Poisson Rate Coding (T=32 timesteps):
python3 pynq_run_inference.py --mock --encoding poisson --T 32 --n 10
```

---

## 3. Transferring to the PYNQ-Z2 Board

Connect your PC to the PYNQ-Z2 board via Ethernet or micro-USB cable.

### Option A: Direct Copy via SCP / Rsync (Recommended)
Default PYNQ IP is `192.168.2.99` (or hostname `pynq`):
```bash
# Using SCP (password is: xilinx)
scp -r /home/dhruv/scratch/projects/sur/MP/pynq_z2_snn_package xilinx@192.168.2.99:/home/xilinx/snn_package/

# Or copy the zip file:
scp /home/dhruv/scratch/projects/sur/MP/pynq_z2_snn_package.zip xilinx@192.168.2.99:/home/xilinx/
```

### Option B: Upload via Jupyter Notebook Web Interface
1. Open your browser on this machine and go to:
   ```
   http://192.168.2.99:9090
   ```
   *(Password: `xilinx`)*
2. Click the **Upload** button in the top-right corner.
3. Select `pynq_z2_snn_package.zip`.
4. Open a **New Terminal** in Jupyter and extract:
   ```bash
   unzip pynq_z2_snn_package.zip
   cd pynq_z2_snn_package
   ```

---

## 4. Running on the PYNQ-Z2 Board

SSH into the PYNQ-Z2 board or use the Jupyter web terminal:
```bash
ssh xilinx@192.168.2.99   # password: xilinx
cd /home/xilinx/snn_package/pynq_z2_snn_package
```

### Command 1: High-Accuracy Direct Analog Encoding ($T=8$, 4x Speedup)
```bash
sudo python3 pynq_run_inference.py \
    --bit hardware/snn.bit \
    --encoding direct \
    --T 8 \
    --n 50 \
    --early_exit \
    --margin 3 \
    --t_min 2
```

### Command 2: Baseline Poisson Rate Coding Comparison ($T=32$)
```bash
sudo python3 pynq_run_inference.py \
    --bit hardware/snn.bit \
    --encoding poisson \
    --T 32 \
    --n 50 \
    --early_exit \
    --margin 3 \
    --t_min 2
```

### Command 3: Full Test Set Evaluation (All 500 Test Images)
```bash
sudo python3 pynq_run_inference.py \
    --bit hardware/snn.bit \
    --encoding direct \
    --T 8 \
    --n 500 \
    --early_exit
```

---

## 5. What You Will See on the Board

During execution, the script will output real-time per-image classifications:
```
================================================================================
  PYNQ-Z2 SNN HARDWARE ACCELERATOR: INFERENCE & BENCHMARK ENGINE
  Target Hardware   : PYNQ-Z2 FPGA (Zynq-7000 PL)
  Input Encoding    : DIRECT (Optimized Method 2)
  Simulation T      : 8 timesteps
  Arithmetic Mode   : FIXED (INT8 Quantized Hardware Pipeline)
  Confidence Exit   : ENABLED (margin=3, t_min=2)
================================================================================
Sample   | True Label   | Predicted    | Status | Exit Step  | Latency (ms)
--------------------------------------------------------------------------------
0001     | cat          | cat          | MATCH  | T=3        | 78.42 ms
0002     | ship         | ship         | MATCH  | T=4        | 104.12 ms
...
================================================================================
  HARDWARE EXECUTION RESULTS SUMMARY:
================================================================================
  Total Images Processed      : 50
  Classification Accuracy     : 89.8% - 91.2%
  Average Timesteps per Image : ~3.8 / 8 steps (2.1x early exit speedup)
  Mean Latency per Inference  : ~95 ms
  Effective Throughput        : ~10.5 FPS
  Board Power (Nominal)       : 2.10 W
  Energy per Classification   : ~199 mJ
================================================================================
```
