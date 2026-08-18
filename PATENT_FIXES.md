# Patent Draft — Fixes Needed

Tracking doc for reconciling the patent draft with the actual project. Update as items
are resolved.

## Document-only fixes (no code changes)
- [ ] **Title**: change to "AI-Based Skin Disease Classification Using Batch Normalization
      and CNN" (currently reads "...Using CNN and Batch Normalization Techniques")
- [ ] **Figure 1**: redraw/replace — currently shows 8 output classes including
      "Squamous Cell Carcinoma," which contradicts the text's "7 categories" and our
      actual `CLASS_MAPPING` (SCC is explicitly excluded). Should show exactly: Melanoma,
      Melanocytic Nevus, Basal Cell Carcinoma, Actinic Keratosis, Benign Keratosis,
      Dermatofibroma, Vascular Lesion.
- [ ] **Working Example / results narrative (Section 8)**: rewrite to match real findings
      instead of the idealized version. Real story: plain fairness training *widened* the
      skin-tone gap and *hurt* Dark-skin accuracy relative to baseline; it only became a
      net improvement once combined with test-time augmentation. Also worth adding: the
      lesion-level data leakage we found and fixed (a genuine, rare methodological
      strength — most published dermatology-AI work doesn't catch this).
- [ ] **Framing check**: "EfficientNet-B4 with batch normalization" is described as if
      novel; EfficientNet-B4 has BatchNorm throughout by default. Our actual addition is
      one `BatchNorm1d` in the classification head. Keep this in mind if an examiner
      pushes on the novelty argument.

## Code fix (chosen path: make Claim 5 true rather than narrow it)
- [x] Implement **joint skin-tone × disease-class weighted focal loss** (`src/models/train_fairness.py`)
      — a 3×7 (Fitzpatrick group × class) inverse-frequency weight matrix, floored at a
      minimum cell count of 20 and capped at 5× the mean weight so tiny cells (e.g.
      Dark-skin Dermatofibroma, single digits of samples) don't destabilize training.
- [x] Retrain baseline + fairness from scratch on **cloud GPU** (Kaggle, Tesla P100) with
      the new loss. Took multiple environment-fix iterations to get working: account
      switch (GPU/internet wouldn't attach on the original account despite correct
      config — cause never fully diagnosed), P100 Pascal architecture unsupported by
      Kaggle's default PyTorch build, then a NumPy 1.x/2.x ABI conflict from the first
      fix attempt. Final working combo: `torch==2.4.1+cu121` / `torchvision==0.19.1`
      (see `kaggle_kernel/`).
- [x] Re-run full evaluation/ablation (test set, MC-Dropout, TTA) on the new models.

### Result: the joint weighting did not achieve its goal

Honest outcome — the fairness gap got *worse*, not better, in every matched comparison
against the old (simple per-group-only weighting) models:

| Variant | Overall | MEL Recall | Light | Medium | Dark | Gap |
|---|---|---|---|---|---|---|
| OLD Fairness | 74.65% | 54.87% | 74.44% | 79.17% | 70.65% | 8.52 |
| OLD Fairness+TTA | 75.43% | 54.13% | 75.25% | 79.55% | 71.74% | **7.81** |
| NEW Fairness | 74.68% | 62.54% | 74.40% | 80.30% | 69.78% | 10.52 |
| NEW Fairness+TTA | 75.08% | 62.83% | 74.62% | 80.49% | 71.52% | 8.97 |

In the new run, plain **baseline+TTA (75.97%, gap 7.18) beats every fairness variant**
on both accuracy and fairness gap — the fairness intervention isn't even the best
option among its own siblings. The one clear improvement, MEL recall jumping from
~50-55% to ~62-63%, shows up in baseline too (which never touches the joint
weighting), so it's attributable to the larger batch size used on cloud GPU (32 vs 8
locally), not to Claim 5's mechanism.

**Decision**: keep the new (joint-weighted) models and honest mixed result on the
`update` branch rather than main, pending a final call on whether to merge. Claim 5 is
now literally true (the code does compute and apply joint per-group×class weights) —
whether to report it as a positive or a negative/mixed finding in the patent write-up
is still open. Old (per-group-only, better fairness gap) models + results are preserved
in `archive_pre_joint_weighting/` for comparison either way.

- [ ] Update the patent's results section with these final real numbers (once the
      keep-vs-revert decision above is made).
- [ ] Update `app.py` dashboard/model paths if the decision changes which checkpoints
      are canonical (currently `src/models/*_best.pt` = the NEW joint-weighted models).

## Staged accuracy-improvement plan (Aug 2026)

Constraint: must stay a CNN with BatchNorm (patent title), so backbone options are
limited to CNNs (no ViT/Swin). Plan: resolution bump -> extra ISIC melanoma data ->
EfficientNetV2-S backbone swap -> ensembling, each stage isolated and reported before
deciding whether to keep it.

### Stage 1: resolution bump 380 -> 456 (fairness model only)

Warm-started the fairness model from the existing, unchanged 380-trained baseline
checkpoint and fine-tuned at 456x456 (FixRes-style — EfficientNet is fully
convolutional until global pooling, so this is architecturally valid). Baseline was
NOT retrained, to isolate the resolution variable.

| Variant | Overall | MEL Recall | Light | Medium | Dark | Gap |
|---|---|---|---|---|---|---|
| Fairness 380 (prior) | 74.68% | 62.54% | 74.40% | 80.30% | 69.78% | 10.52 |
| Fairness 380+TTA (prior) | 75.08% | 62.83% | 74.62% | 80.49% | 71.52% | 8.97 |
| **Fairness 456 (Stage 1)** | **75.70%** | **59.59%** | 75.47% | 80.49% | 71.52% | 8.97 |
| **Fairness 456+TTA (Stage 1)** | **76.75%** | **61.50%** | 76.49% | 81.43% | 72.83% | 8.60 |
| (reference) Baseline 380+TTA | 75.97% | 62.83% | 76.42% | 78.05% | 70.87% | 7.18 |

Mixed result: overall accuracy and fairness gap both improved (Fairness 456+TTA is now
the best single model overall, 76.75%, beating the previous best baseline+TTA at
75.97%), but **melanoma recall dropped** (62.83% -> 61.50% with TTA, 62.54% -> 59.59%
without) — the opposite direction from the metric we're actually trying to move.
Decision on whether to keep the resolution bump pending user review.
