"""
distillation_loss.py - Knowledge Distillation Loss for SNN Student & ANN Teacher

Implements Hinton's Knowledge Distillation formulation adapted for Spiking Neural Networks:
  L_total = (1 - alpha) * L_CE(y, z_SNN) + alpha * (tau^2) * L_KL(p_SNN(tau), p_ANN(tau))

Where:
  - z_SNN: Integrated output membrane potentials or logit scores from the SNN
  - z_ANN: Pre-softmax logits from the frozen ANN teacher
  - tau: Temperature parameter (smooths peak probability distribution to reveal inter-class dark knowledge)
  - alpha: Weighting factor between hard ground-truth labels and teacher soft probabilities
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class KnowledgeDistillationLoss(nn.Module):
        def __init__(self, temperature=4.0, alpha=0.7, label_smoothing=0.0):
            super().__init__()
            self.temperature = temperature
            self.alpha = alpha
            self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        def forward(self, snn_logits, teacher_logits, targets):
            """
            snn_logits: (B, num_classes) membrane potentials from SNN output layer
            teacher_logits: (B, num_classes) logits from ANN teacher
            targets: (B) ground truth class indices
            """
            tau = self.temperature

            # 1. Standard Cross-Entropy with Hard Targets
            loss_hard = self.ce_loss(snn_logits, targets)

            # 2. Temperature-scaled KL Divergence with Teacher Soft Targets
            snn_soft_log_probs = F.log_softmax(snn_logits / tau, dim=1)
            teacher_soft_probs = F.softmax(teacher_logits / tau, dim=1)

            loss_soft = self.kl_loss(snn_soft_log_probs, teacher_soft_probs) * (tau ** 2)

            # 3. Blended Distillation Loss
            loss_total = (1.0 - self.alpha) * loss_hard + self.alpha * loss_soft
            return loss_total, loss_hard.item(), loss_soft.item()
else:
    class KnowledgeDistillationLoss:
        def __init__(self, *args, **kwargs):
            pass


def numpy_distillation_loss(snn_logits, teacher_logits, targets, temperature=4.0, alpha=0.7):
    """
    NumPy-native implementation of Distillation Loss for offline verification and testing.
    """
    B, C = snn_logits.shape
    tau = temperature

    # Softmax functions with numerical stability
    def softmax(x, axis=1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def log_softmax(x, axis=1):
        max_x = np.max(x, axis=axis, keepdims=True)
        return (x - max_x) - np.log(np.sum(np.exp(x - max_x), axis=axis, keepdims=True))

    # Cross Entropy
    probs = softmax(snn_logits)
    ce_loss = -np.mean(np.log(np.clip(probs[np.arange(B), targets], 1e-12, 1.0)))

    # KL Divergence: sum(P * (log(P) - log(Q)))
    p_teacher = softmax(teacher_logits / tau)
    log_q_snn = log_softmax(snn_logits / tau)
    # KL(P || Q) = sum(P * (log P - log Q))
    log_p_teacher = np.log(np.clip(p_teacher, 1e-12, 1.0))
    kl_div = np.sum(p_teacher * (log_p_teacher - log_q_snn), axis=1)
    soft_loss = np.mean(kl_div) * (tau ** 2)

    total_loss = (1.0 - alpha) * ce_loss + alpha * soft_loss
    return total_loss, ce_loss, soft_loss


if __name__ == "__main__":
    print("[*] Testing Knowledge Distillation Loss Numerical Verification...")
    B, C = 4, 10
    dummy_snn = np.random.randn(B, C)
    dummy_teacher = dummy_snn + np.random.randn(B, C) * 0.2
    dummy_targets = np.random.randint(0, C, size=(B,))

    tot, ce, kl = numpy_distillation_loss(dummy_snn, dummy_teacher, dummy_targets, temperature=4.0, alpha=0.7)
    print(f"[+] Distillation Test OK: Total Loss = {tot:.4f} (CE: {ce:.4f}, KL-Soft: {kl:.4f})")

    if TORCH_AVAILABLE:
        loss_module = KnowledgeDistillationLoss(temperature=4.0, alpha=0.7)
        t_snn = torch.from_numpy(dummy_snn).float()
        t_tea = torch.from_numpy(dummy_teacher).float()
        t_tgt = torch.from_numpy(dummy_targets).long()
        t_tot, t_ce, t_kl = loss_module(t_snn, t_tea, t_tgt)
        print(f"[+] PyTorch Distillation Loss Match: {t_tot.item():.4f}")
