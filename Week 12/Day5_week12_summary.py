summary = """
╔══════════════════════════════════════════════════════════════╗
║                   WEEK 12 SUMMARY                                                                    ║
╚══════════════════════════════════════════════════════════════╝

Day 1: Image CLIP embeddings extracted
  1654 training + 200 test concept embeddings
  From actual THINGS images (multi-image average)
  Discriminability: 0.529 (vs text 0.713)

Day 2: ATM retrained — 3 strategies compared
  Text CLIP:       0.035 (1.4×) — baseline
  Mixed 70/30:     0.030 (1.2×) — mixing hurts
  Image CLIP T=0.05: 0.045 (1.8×) — BEST
  Image CLIP T=0.03: 0.040 (1.6×) — too hard

Day 3: Full evaluation of best model
  Median rank: 100/200 — at chance overall
  Above-chance signal for easy concepts
  P3 most important: parietal/P300 attention
  baton→blowtorch: shape captured, not semantics
  Generation pipeline working end-to-end

Day 4: 63-channel ATM
  Top-5: 0.035 — lower than 17-ch (0.045)
  Loss diverged — overfitting (1654 samples)
  Temporal region most important: inferior
  temporal cortex → object identity pathway
  Conclusion: data scale is the bottleneck

Day 5: Paper 2 written
  Full IEEE-format paper draft
  4 pages, 1 table, methods + results + discussion
  Ready for Overleaf upload

PHASE 2 STATUS AT END OF WEEK 12:
  ✅ Data pipeline: 47.5GB → (1654, 17, 100)
  ✅ CLIP targets: image embeddings computed
  ✅ ATM encoder: 626K params, trained
  ✅ Best result: Top-5 = 0.045 (1.8× chance)
  ✅ Generation: EEG → SD → image (working)
  ✅ Interpretability: channel attention maps
  ✅ Paper 2: draft complete

WHAT REMAINS FOR PHASE 2 COMPLETION:
  ⬜ Individual-subject training (10× more data)
  ⬜ GPU training (larger batches → better signal)
  ⬜ Muse S headband (real REM EEG)
  ⬜ Prospective self-experiment
  ⬜ Submit paper to NeurIPS ML4H (Oct deadline)

PRIMARY BOTTLENECK IDENTIFIED:
  1654 averaged pairs vs 16,540 individual pairs
  (Scotti et al. used individual not averaged)
  Fix: train on each subject separately
  Impact: 10× more training data, same model
  Estimated improvement: 0.045 → 0.10-0.15 Top-5
"""
print(summary)