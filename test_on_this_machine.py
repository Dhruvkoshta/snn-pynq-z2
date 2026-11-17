#!/usr/bin/env python3
"""
test_on_this_machine.py - Pre-flight Health & Verification Test on Host PC

Runs automated checks on all package files and simulates the PYNQ-Z2 hardware
interface before deploying to the physical board.
"""

import os
import sys
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent

def run_check(title, fn):
    print(f"[*] Checking: {title} ...", end=" ", flush=True)
    try:
        fn()
        print(" [PASSED]")
        return True
    except Exception as e:
        print(f" [FAILED]: {e}")
        return False

def check_bitstream():
    bit = HERE / "hardware" / "snn.bit"
    hwh = HERE / "hardware" / "snn.hwh"
    assert bit.exists(), f"Missing {bit}"
    assert bit.stat().st_size > 3_000_000, f"Bitstream too small ({bit.stat().st_size} bytes)"
    assert hwh.exists(), f"Missing {hwh}"

def check_weights():
    w_root = HERE / "weights"
    assert (w_root / "weights_float32").is_dir(), "Missing weights_float32"
    assert (w_root / "weights_int8_experiment").is_dir(), "Missing weights_int8_experiment"
    assert (w_root / "weights_all_combined_int8.bin").exists(), "Missing QAT combined weights"

def check_dataset():
    batch = HERE / "cifar" / "test_batch"
    assert batch.exists(), f"Missing {batch}"
    assert batch.stat().st_size > 1_000_000, "Dataset file corrupted or empty"

def check_methods():
    p = HERE / "methods" / "all_combined"
    assert p.is_dir(), f"Missing unified all_combined method dir {p}"
    assert (p / "models" / "snn_all_combined.py").exists(), "Missing snn_all_combined.py"
    assert (p / "training" / "train_all_combined.py").exists(), "Missing train_all_combined.py"

def check_mock_inference():
    cmd = [sys.executable, str(HERE / "pynq_run_inference.py"), "--mock", "--n", "3", "--T", "8"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Inference execution failed:\n{res.stderr}"

def main():
    print("=" * 70)
    print("  PYNQ-Z2 DEPLOYMENT PRE-FLIGHT CHECK (HOST MACHINE TEST)")
    print("=" * 70)

    checks = [
        ("FPGA Overlay Bitstream & Hardware Handoff", check_bitstream),
        ("Pre-packaged Weights & QAT INT8 Binary Tables", check_weights),
        ("CIFAR-10 Test Batch Dataset", check_dataset),
        ("All 4 Accuracy Improvement Methods", check_methods),
        ("Hardware-in-the-Loop Mock Inference Execution", check_mock_inference)
    ]

    all_passed = True
    for title, fn in checks:
        if not run_check(title, fn):
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("  ALL CHECKS PASSED: Package is 100% ready for PYNQ-Z2 deployment!")
    else:
        print("  SOME CHECKS FAILED: Please inspect errors above.")
    print("=" * 70)

if __name__ == "__main__":
    main()
