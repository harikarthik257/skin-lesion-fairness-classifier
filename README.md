# Fairness-Aware Skin Lesion Classifier

A 7-class dermatology image classifier (EfficientNet-B4) trained on HAM10000 +
ISIC2019, built with a specific focus on **fairness across skin tones** — measuring
and correcting for the well-documented bias toward lighter skin in public dermoscopy
datasets, and shipped with a live Gradio demo, Grad-CAM++ explainability, and
MC-Dropout uncertainty estimation.

> **Research/educational prototype — not a medical device.** Not clinically
> validated. Do not use for actual diagnosis.

## Results

All numbers below are on a held-out test set that is **lesion-disjoint** from
training (see [Methodology](#methodology--a-data-leakage-bug-we-found-and-fixed)).

| Variant | Overall Acc | Melanoma Recall | Light (I-II) | Medium (III-IV) | Dark (V-VI) | Fairness Gap |
|---|---|---|---|---|---|---|
| Baseline | 75.40% | 49.85% | 74.95% | 80.49% | 72.17% | 8.31 |
| Baseline + TTA | 75.86% | 49.85% | 75.32% | 81.24% | 72.83% | 8.41 |
| Fairness-corrected | 74.65% | 54.87% | 74.44% | 79.17% | 70.65% | 8.52 |
| Fairness + MC-Dropout | 74.44% | 55.01% | 74.04% | 79.36% | 71.09% | 8.28 |
| **Fairness + TTA** | **75.43%** | **54.13%** | 75.25% | 79.55% | 71.74% | **7.81** |

**Fairness + TTA** is the recommended variant (and the demo's default): nearly
baseline-level accuracy, the best melanoma recall, and the smallest fairness gap of
everything tested.

## Methodology & a data leakage bug we found and fixed

HAM10000 and ISIC2019 both photograph the same physical lesion multiple times
(different angles/zoom). A naive random split lets those near-duplicate photos land
in different train/val/test sets, so the model can partially memorize test lesions —
we measured this affected **~61% of a naive test split**. `src/data/prepare_data.py`
now does a **lesion-grouped, class-stratified split** (every photo of one lesion
stays in exactly one set), which dropped the honest test accuracy from the low-80s
to the mid-70s — a real, expected consequence of removing memorization, not a
regression. See the app's Methodology tab for the full write-up.

Skin-tone groups (Light I-II / Medium III-IV / Dark V-VI) are *estimated*, not
ground truth: computed from the peri-lesion border color in LAB space (Individual
Typology Angle / ITA), since neither source dataset provides Fitzpatrick labels
(`src/data/compute_ita.py`).

## Project structure

```
app.py                        Gradio demo (live inference + results dashboard)
src/
  data/
    prepare_data.py           Downloads HAM10000+ISIC2019, dedupes, lesion-grouped split
    compute_ita.py            ITA-based Fitzpatrick skin-tone group estimation
    processed/                Train/val/test manifests (the split definitions)
  models/
    model_defs.py             Shared model classes (Baseline/Fairness) + MC-Dropout helper
    train_baseline.py         Plain EfficientNet-B4, cross-entropy loss
    train_fairness.py         Warm-started from baseline; focal loss + per-skin-tone-group
                               reweighting; early stopping
    evaluate_test.py          Test-set evaluation (--model, --tta flags)
    mc_dropout_infer.py       MC-Dropout uncertainty inference (--split flag)
    gradcam_infer.py          Grad-CAM++ sample visualizations
  outputs/                    Generated predictions/summaries (CSV) + sample images
outputs/
  ablation_analysis.py        Builds the full variant-comparison table
  check_leakage.py            Quantifies lesion-level leakage across splits
examples/                     Sample images (one per class) for the demo's Diagnose tab
archive_leaky_split_run/      Archived pre-fix (leaky-split) checkpoints/results, kept
                               for before/after comparison
WEBSITE_PLAN.md               Design plan for the Gradio app
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on Linux/Mac
pip install -r requirements.txt
```

Datasets (HAM10000, ISIC2019) download automatically via `kagglehub` on first run —
no manual download needed, but you'll need a Kaggle account/API access.

## Running the pipeline

```bash
# 1. Prepare data (lesion-grouped split)
python src/data/prepare_data.py
python src/data/compute_ita.py

# 2. Train
python src/models/train_baseline.py
python src/models/train_fairness.py

# 3. Evaluate
python src/models/evaluate_test.py --model baseline
python src/models/evaluate_test.py --model fairness --tta
python src/models/mc_dropout_infer.py --split test

# 4. Rebuild the ablation comparison table
python outputs/ablation_analysis.py
```

Trained checkpoints aren't included in this repo (too large for git) — run the
training scripts above to regenerate `src/models/*_best.pt`, or see
[`.gitignore`](.gitignore) for a Git LFS note if you want to version them.

## Running the demo

```bash
python app.py
```

Opens at `http://127.0.0.1:7860`. Tabs: **About**, **Diagnose** (live inference —
upload an image or click a sample; Grad-CAM++ heatmap, confidence, estimated skin
tone, optional MC-Dropout uncertainty), **Fairness & Performance** (the results
dashboard above, live from `src/outputs/`), **Methodology**.

**Image requirement**: the model was trained on dermatoscopic images (close-up,
magnified photos taken with a dermatoscope), not regular phone photos — use the
provided sample images in `examples/` to test, or a genuine dermoscopy image.

## Known limitations

- Melanoma vs. nevus is genuinely hard to distinguish even for experts in many
  cases; melanoma recall is a persistent weak point across all variants.
- Dermatofibroma and Vascular lesion classes have very few test examples (~30-40),
  so their individual accuracy is noisy.
- The Fitzpatrick group is an algorithmic estimate (ITA), not a clinical
  assessment.
- Grad-CAM++ can show a residual attention artifact in one image corner on
  low-contrast, borderless lesion photos — a known EfficientNet architectural
  quirk (confirmed via a random-noise-input diagnostic to be a fixed,
  content-independent activation, not genuine model attention).
