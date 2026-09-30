import numpy as np
import os

THINGS_DIR = (r"C:\Users\Hp\.vscode\PROJECT 07"
              r"\classifier_main_pipeline\data"
              r"\things_eeg")

print("=== Phase 2 Complete Results Summary ===\n")
print("For Paper 2 methods and results sections.\n")

# Dataset
eeg_train = np.load(os.path.join(
    THINGS_DIR, 'eeg_train_avg_all_subjects.npy'))
eeg_test  = np.load(os.path.join(
    THINGS_DIR, 'eeg_test_avg_all_subjects.npy'))
clip_img  = np.load(os.path.join(
    THINGS_DIR, 'clip_IMAGE_targets_train.npy'))

print("DATASET:")
print(f"  Subjects averaged: ~10 (from 47.5GB download)")
print(f"  Training EEG: {eeg_train.shape}"
      f"  → (1654 concepts, 17 ch, 100 tp)")
print(f"  Test EEG:     {eeg_test.shape}"
      f"  → (200 concepts, 17 ch, 100 tp)")
print(f"  Sampling:     100 Hz, window -200ms to 790ms")
print(f"  CLIP targets: {clip_img.shape}"
      f"  → image embeddings, mean sim=0.529")
print()

print("EXPERIMENTS (all on 200 test concepts):")
experiments = [
    ("ATM-17  Text-CLIP  T=0.07", 0.035, 1.4),
    ("ATM-17  Mixed-70   T=0.05", 0.030, 1.2),
    ("ATM-17  Img-CLIP   T=0.05", 0.045, 1.8),
    ("ATM-17  Img-CLIP   T=0.03", 0.040, 1.6),
    ("ATM-63  Img-CLIP   T=0.05", 0.035, 1.4),
]
print(f"  {'Model':35s} {'Top-5':>6} {'×chance':>8}")
print(f"  {'Chance':35s} {'0.025':>6} {'1.0×':>8}")
print("  " + "-"*52)
for name, t5, mult in experiments:
    flag = " ← BEST" if t5 == 0.045 else ""
    print(f"  {name:35s} {t5:.3f}  {mult:.1f}×{flag}")

print()
print("KEY FINDINGS:")
findings = [
    "Image > Text CLIP targets (1.8× vs 1.4× chance)",
    "Mixed targets hurt: text adds noise, not signal",
    "Standard T=0.05 > harder T=0.03",
    "17-ch > 63-ch: data bottleneck, not architecture",
    "P3 most important in 17-ch: parietal attention",
    "Temporal most important in 63-ch: object identity",
    "baton4→blowtorch: EEG captures shape, not semantics",
    "Median rank=100: chance overall, above-chance subset",
    "Generation pipeline: end-to-end working (ComfyUI)",
]
for i, f in enumerate(findings):
    print(f"  {i+1}. {f}")