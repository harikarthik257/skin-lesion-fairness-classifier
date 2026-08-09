# Website / Demo Plan

Status: PLANNING ONLY — nothing built yet. Written so implementation can resume later
without re-deriving the design. No GPU work happens until we start Phase 2 below.

## Goal

A Gradio web app with two purposes:
1. Let someone upload a skin lesion photo and get a prediction, a confidence score,
   an uncertainty estimate, and a Grad-CAM++ heatmap explaining the prediction.
2. Present the fairness story of the project (baseline vs fairness-corrected model,
   per-skin-tone accuracy, the data-leakage fix) as a readable dashboard, not just
   buried in CSVs.

Framework: **Gradio** (`gr.Blocks`, multi-tab), already in requirements.txt. Runs
locally to start; can consider Hugging Face Spaces for public hosting later (would
need a CPU-inference fallback path since EfficientNet-B4 is slow on CPU — a later
concern, not solved in this plan).

## Tab structure

### 1. Home / About
- Plain-language project description: 7-class skin lesion classifier (MEL, NV, BCC,
  AKIEC, BKL, DF, VASC), trained on HAM10000 + ISIC2019.
- The fairness angle: why skin-tone bias matters in dermatology AI, what we did about
  it (Fitzpatrick-group-weighted focal loss).
- **Prominent disclaimer**: research/educational prototype, NOT a medical device, not
  a diagnosis, always consult a dermatologist. Required for anything health-related.

### 2. Diagnose (the interactive demo)
- Input: image upload (`gr.Image`)
- Controls:
  - Model selector: Baseline vs Fairness (lets someone directly compare predictions
    side by side — good demo moment)
  - Toggle: single-pass (fast) vs MC-Dropout 30-pass (slower, ~200ms/image, gives an
    uncertainty estimate) — surfaces the latency/uncertainty tradeoff we already
    quantified in the ablation study instead of hiding it
- Output:
  - Predicted class (human-readable label, e.g. "Melanoma (MEL)") + confidence %
  - Bar chart of all 7 class probabilities, not just the top-1 (more honest than a
    single number, especially for visually similar classes like MEL/NV)
  - Grad-CAM++ heatmap overlay (using the `blocks[-1]` target layer fix, not the
    broken `conv_head` one)
  - If MC-Dropout is on: entropy/variance shown, with a plain-language flag like
    "high uncertainty — low confidence in this result" past a threshold
  - Estimated Fitzpatrick skin-tone group (via ITA), shown for transparency about
    what the fairness mechanism is reacting to — not presented as a diagnostic input
- Gradio's built-in request queue prevents multiple simultaneous users from fighting
  over the single GPU.

### 3. Fairness & Performance (dashboard, static — no GPU, reads precomputed CSVs)
- The ablation table (baseline vs fairness vs fairness+MC-Dropout): overall accuracy,
  MEL recall, per-Fitzpatrick-group accuracy, fairness gap.
- Bar chart: per-group accuracy, baseline vs fairness side by side (visually makes
  the "fairness training helped the worst-off group" point immediately, rather than
  requiring someone to read a table).
- Confusion matrix.
- Calibration: entropy separation between correct/incorrect predictions.
- This tab is just rendering already-computed results (from `outputs/ablation_analysis.py`
  and `evaluate_test.py` outputs) — can be built without touching the GPU at all.

### 4. Methodology / Data Card
- Datasets (HAM10000 + ISIC2019), class definitions, train/val/test sizes.
- **The data leakage bug and fix**, told plainly — this is a genuine strength of the
  project (caught and fixed a real methodological flaw) and worth surfacing, not
  hiding.
- Known limitations: Grad-CAM's residual corner-artifact on low-contrast/borderless
  images, ITA-based Fitzpatrick grouping is an estimate not ground truth, small class
  sizes for DF/VASC, fairness-gap direction wasn't stable between val and test in the
  original (leaky) run.

## Technical plan

- New file: `app.py` at repo root (Gradio/HF Spaces convention — makes future
  deployment simpler if that's ever wanted).
- **Consolidate model class definitions.** Right now `BaselineModel`/`FairnessModel`
  are copy-pasted across `train_baseline.py`, `train_fairness.py`, `evaluate_test.py`,
  `mc_dropout_infer.py`, and `gradcam_infer.py`. Before building the app, pull them
  into one shared `src/models/model_defs.py` and have the app (and ideally the
  existing scripts) import from there — avoids a 6th copy and future drift between
  them.
- Reuse `compute_ita.py`'s `compute_ita`/`assign_fitzpatrick` for the live
  skin-tone estimate in the Diagnose tab.
- Reuse `gradcam_infer.py`'s Grad-CAM++ setup, specifically the `blocks[-1]` target
  layer (the `conv_head` corner-artifact bug must not make it into the app).
- Load both model checkpoints once at app startup (module-level, cached), not per
  request.
- Model checkpoint paths as named constants, easy to swap once fairness retraining
  (currently paused, mid-epoch-1 progress preserved in
  `models/fairness_efficientnet_b4_best.pt`) finishes and produces a final checkpoint.

## Phasing

- **Phase 1 (can start now, zero GPU cost):** scaffold `app.py`, build the Home/About
  and Methodology tabs (static content), build the Fairness & Performance dashboard
  tab (reads existing CSVs only), consolidate model class definitions into
  `model_defs.py`.
- **Phase 2 (needs GPU, do after fairness retraining resumes/finishes):** wire up the
  Diagnose tab's live inference (prediction, Grad-CAM++, MC-Dropout), test end-to-end
  with real uploads, swap in the final fairness checkpoint.

## Open questions for later

- Public hosting (HF Spaces) or local-only? Affects whether we need a CPU-inference
  fallback.
- Should the fairness-vs-baseline model selector be visible to end users, or should
  the app just always use the fairness model in a simple "production" mode, with the
  comparison view reserved for the Fairness & Performance dashboard?
