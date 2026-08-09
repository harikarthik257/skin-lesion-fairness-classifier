"""
Gradio app for the skin lesion classification + fairness project.

Phase 1: static tabs -- About, Fairness & Performance dashboard (reads precomputed
CSVs from src/outputs/), and Methodology. No GPU needed.

Phase 2: live inference on the Diagnose tab -- upload an image, get a prediction,
confidence, class-probability breakdown, Grad-CAM++ heatmap, estimated skin-tone
group, and an optional MC-Dropout uncertainty estimate. Default inference mode is
the fairness-corrected model + test-time augmentation (found in the ablation study
to give the best balance of accuracy, melanoma recall, and fairness gap).
See WEBSITE_PLAN.md for the full design.
"""
import os
import sys
import glob
import cv2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr
import torch
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "src", "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "src", "models")

MODELS_SRC_DIR = os.path.join(BASE_DIR, "src", "models")
if MODELS_SRC_DIR not in sys.path:
    sys.path.insert(0, MODELS_SRC_DIR)
from model_defs import FairnessModel, enable_dropout, INV_CLASS_MAPPING  # noqa: E402

from pytorch_grad_cam import GradCAMPlusPlus  # noqa: E402
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # noqa: E402
from pytorch_grad_cam.utils.image import show_cam_on_image  # noqa: E402

CLASS_NAMES = {
    'MEL': 'Melanoma', 'NV': 'Melanocytic nevus', 'BCC': 'Basal cell carcinoma',
    'AKIEC': "Actinic keratosis / Bowen's disease", 'BKL': 'Benign keratosis',
    'DF': 'Dermatofibroma', 'VASC': 'Vascular lesion',
}
GROUP_ORDER = ['Light (I-II)', 'Medium (III-IV)', 'Dark (V-VI)']

# --- Design tokens (validated categorical/status palette; see dataviz skill) ----
CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']  # fixed order, never cycled
STATUS = {'good': '#0ca30c', 'warning': '#fab219', 'serious': '#ec835a', 'critical': '#d03b3b'}
INK = {'primary': '#0b0b0b', 'secondary': '#52514e', 'muted': '#898781'}
CHART_SURFACE = '#fcfcfb'
GRIDLINE = '#e1e0d9'

CUSTOM_CSS = """
:root {
  --brand-blue: #2a78d6;
  --brand-blue-dark: #184f95;
  --surface-1: #fcfcfb;
  --border-hairline: rgba(11,11,11,0.10);
  /* Token scale from the design-system skill (primitive-tokens.md) -- kept as
     plain CSS custom properties since this app is Gradio/HTML, not Tailwind. */
  --radius-lg: 0.75rem;
  --radius-2xl: 1rem;
  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.08), 0 2px 4px -2px rgb(0 0 0 / 0.06);
  --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.08), 0 4px 6px -4px rgb(0 0 0 / 0.06);
  --duration-fast: 150ms;
  --duration-normal: 200ms;
  --ring-color: #2a78d6;
}

.hero-banner {
  background: linear-gradient(135deg, #2a78d6 0%, #184f95 100%);
  color: #ffffff;
  padding: 2.25rem 2rem;
  border-radius: var(--radius-2xl);
  margin-bottom: 1.25rem;
  box-shadow: var(--shadow-lg);
}
.hero-banner h1 { font-size: 1.85rem; margin: 0 0 0.4rem 0; font-weight: 700; }
.hero-banner p { font-size: 1.02rem; opacity: 0.92; margin: 0; max-width: 60rem; }
.hero-badges { margin-top: 0.9rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
.hero-badge {
  display: inline-flex; align-items: center; gap: 0.35rem;
  background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.28);
  padding: 0.25rem 0.7rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
  transition: background-color var(--duration-fast) ease-in-out;
}
.hero-badge:hover { background: rgba(255,255,255,0.26); }

.stat-tile-row { display: flex; gap: 0.9rem; flex-wrap: wrap; margin-bottom: 1.1rem; }
.stat-tile {
  flex: 1; min-width: 155px;
  background: var(--surface-1);
  border: 1px solid var(--border-hairline);
  border-radius: var(--radius-lg);
  padding: 0.9rem 1.1rem;
  box-shadow: var(--shadow-sm);
  transition: box-shadow var(--duration-normal) ease-out, transform var(--duration-normal) ease-out;
}
.stat-tile:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); }
.stat-tile .stat-label {
  font-size: 0.72rem; color: #898781; text-transform: uppercase;
  letter-spacing: 0.04em; font-weight: 600;
}
.stat-tile .stat-value { font-size: 1.55rem; font-weight: 700; color: #0b0b0b; margin-top: 0.15rem; }
.stat-tile .stat-sub { font-size: 0.78rem; color: #52514e; margin-top: 0.1rem; }
.stat-tile.stat-highlight { border-color: var(--brand-blue); border-width: 2px; }

.result-card {
  border-radius: var(--radius-lg);
  padding: 1.1rem 1.35rem;
  border: 1px solid var(--border-hairline);
  background: var(--surface-1);
  margin-bottom: 0.6rem;
  box-shadow: var(--shadow-sm);
  transition: box-shadow var(--duration-normal) ease-out;
}
.result-card:hover { box-shadow: var(--shadow-md); }
.result-title { font-size: 1.25rem; font-weight: 700; color: #0b0b0b; margin-bottom: 0.3rem; }
.result-badge {
  display: inline-flex; align-items: center; gap: 0.35rem;
  padding: 0.25rem 0.7rem; border-radius: 999px;
  font-weight: 600; font-size: 0.82rem; margin-top: 0.2rem;
}
.badge-good { background: rgba(12,163,12,0.12); color: #0a7d0a; }
.badge-warning { background: rgba(250,178,25,0.18); color: #8a5a00; }
.badge-critical { background: rgba(208,59,59,0.12); color: #b53232; }
.result-sub { font-size: 0.85rem; color: #52514e; margin-top: 0.4rem; }

/* Focus-visible ring spec (states-and-variants.md) -- keyboard-accessible focus
   indicator on interactive elements, not relying on the browser default outline. */
button:focus-visible, a:focus-visible, [tabindex]:focus-visible {
  outline: none !important;
  box-shadow: 0 0 0 2px var(--surface-1), 0 0 0 4px var(--ring-color) !important;
  transition: box-shadow var(--duration-fast) ease-in-out;
}
"""

# --- Live inference setup -------------------------------------------------------
# Fairness + TTA was found (in the ablation study) to give the best balance of
# accuracy, melanoma recall, and fairness gap among all tested variants, so it's
# the default. MC-Dropout is offered as an alternate mode for an uncertainty
# estimate, at a real latency cost (~30x slower).

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

_MODEL = None
_CAM = None
_MODEL_LOAD_ERROR = None


def _get_model():
    """Lazily load the fairness model + Grad-CAM++ wrapper on first use, so
    static tabs (About, Dashboard, Methodology) work even if this fails."""
    global _MODEL, _CAM, _MODEL_LOAD_ERROR
    if _MODEL is not None or _MODEL_LOAD_ERROR is not None:
        return _MODEL, _CAM, _MODEL_LOAD_ERROR
    try:
        model = FairnessModel(num_classes=7).to(DEVICE)
        ckpt_path = os.path.join(MODELS_DIR, "fairness_efficientnet_b4_best.pt")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        model.eval()
        # blocks[-1], not conv_head -- conv_head has a fixed, content-independent
        # corner-activation artifact (confirmed via a random-noise-input diagnostic)
        # that swamps the real Grad-CAM signal. See WEBSITE_PLAN.md / Methodology tab.
        cam = GradCAMPlusPlus(model=model, target_layers=[model.backbone.blocks[-1]])
        _MODEL, _CAM = model, cam
    except Exception as e:  # noqa: BLE001
        _MODEL_LOAD_ERROR = str(e)
    return _MODEL, _CAM, _MODEL_LOAD_ERROR


def _estimate_fitzpatrick(pil_image):
    """Same ITA-based estimate used to label the training data (src/data/compute_ita.py),
    reimplemented here to work directly on an RGB PIL image instead of a BGR file read."""
    rgb = np.array(pil_image.convert('RGB'))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32) * 100.0 / 255.0
    b = lab[:, :, 2].astype(np.float32) - 128.0

    h, w = L.shape
    my, mx = int(h * 0.15), int(w * 0.15)
    margin_mask = np.zeros((h, w), dtype=bool)
    margin_mask[:my, :] = True
    margin_mask[-my:, :] = True
    margin_mask[:, :mx] = True
    margin_mask[:, -mx:] = True

    valid_mask = margin_mask & (L > 30.0)
    if not np.any(valid_mask):
        valid_mask = margin_mask
    if not np.any(valid_mask):
        valid_mask = np.ones(L.shape, dtype=bool)

    ita = np.degrees(np.arctan2(np.mean(L[valid_mask]) - 50, np.mean(b[valid_mask])))
    if ita > 55:
        return 'Light (I-II)', ita
    elif 28 < ita <= 55:
        return 'Medium (III-IV)', ita
    return 'Dark (V-VI)', ita


def _prob_bar_chart(probs, pred_idx):
    # probs is already ordered 0..6, matching INV_CLASS_MAPPING's integer keys
    ordered_labels = [INV_CLASS_MAPPING[i] for i in range(len(INV_CLASS_MAPPING))]
    values = [probs[i] * 100 for i in range(len(INV_CLASS_MAPPING))]
    # Predicted class in brand blue (identity highlight); others muted -- not a
    # categorical multi-series chart, so a single highlight hue is correct here.
    colors = [CATEGORICAL[0] if i == pred_idx else '#c3c2b7' for i in range(len(ordered_labels))]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    _style_axes(ax, fig)
    bars = ax.barh(ordered_labels, values, color=colors)
    for bar, val in zip(bars, values):
        if val > 3:
            ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2, f'{val:.1f}%',
                    va='center', fontsize=8, color=INK['secondary'])
    ax.set_xlabel('Probability (%)')
    ax.set_xlim(0, 108)
    ax.invert_yaxis()
    ax.grid(axis='x', color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def _confidence_badge(confidence):
    if confidence >= 80:
        return '<span class="result-badge badge-good">&#10003; High confidence</span>'
    elif confidence >= 50:
        return '<span class="result-badge badge-warning">&#9888; Moderate confidence</span>'
    return '<span class="result-badge badge-critical">&#9888; Low confidence -- verify with a specialist</span>'


def predict(image, mode):
    empty_card = '<div class="result-card"><span class="result-sub">Upload an image first.</span></div>'
    if image is None:
        return None, None, empty_card, "", ""

    model, cam, load_error = _get_model()
    if load_error is not None:
        error_card = f'<div class="result-card"><span class="result-sub">Model failed to load: {load_error}</span></div>'
        return None, None, error_card, "", ""

    pil_image = image.convert('RGB')
    resized_for_display = pil_image.resize((380, 380))
    tensor = INFER_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)

    uncertainty_text = ""
    if mode == "Uncertainty-aware (MC-Dropout, ~30x slower)":
        enable_dropout(model)
        with torch.no_grad():
            batch = tensor.repeat(30, 1, 1, 1)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                probs_all = torch.softmax(model(batch), dim=1)
        mean_probs = probs_all.mean(dim=0)
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-9)).item()
        model.eval()  # restore eval mode (disables dropout again) for the Grad-CAM pass below
        if entropy > 0.8:
            badge = '<span class="result-badge badge-critical">&#9888; High uncertainty -- recommend clinical follow-up</span>'
        else:
            badge = '<span class="result-badge badge-good">&#10003; Low uncertainty</span>'
        uncertainty_text = (f'<div class="result-card"><b>Predictive entropy:</b> {entropy:.3f} '
                             f'(from 30 stochastic passes) {badge}</div>')
    else:  # Fast (TTA) -- the default, recommended mode
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                views = [tensor, torch.flip(tensor, dims=[-1]), torch.flip(tensor, dims=[-2])]
                probs_sum = sum(torch.softmax(model(v), dim=1) for v in views)
        mean_probs = (probs_sum / len(views))[0]

    mean_probs_np = mean_probs.detach().cpu().numpy()
    pred_idx = int(mean_probs_np.argmax())
    pred_code = INV_CLASS_MAPPING[pred_idx]
    pred_label = f"{CLASS_NAMES[pred_code]} ({pred_code})"
    confidence = mean_probs_np[pred_idx] * 100

    # Grad-CAM++ -- always a standard single deterministic forward+backward pass,
    # independent of whether MC-Dropout was used for the displayed prediction.
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0, :]
    rgb_float = np.float32(resized_for_display) / 255.0
    heatmap = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    prob_chart = _prob_bar_chart(mean_probs_np, pred_idx)
    fitz_group, ita = _estimate_fitzpatrick(pil_image)
    fitz_text = (f'<div class="result-card"><b>Estimated skin-tone group:</b> {fitz_group} '
                 f'<span class="result-sub">(ITA={ita:.1f}° -- algorithmic estimate for '
                 f'transparency, not a clinical assessment)</span></div>')

    prediction_html = (
        f'<div class="result-card">'
        f'<div class="result-title">{pred_label}</div>'
        f'{_confidence_badge(confidence)}'
        f'<div class="result-sub">{confidence:.1f}% confidence '
        f'({"TTA-averaged" if mode != "Uncertainty-aware (MC-Dropout, ~30x slower)" else "MC-Dropout-averaged"})</div>'
        f'</div>'
    )
    return heatmap, prob_chart, prediction_html, fitz_text, uncertainty_text


# Display name -> predictions CSV, covering every variant from the ablation study.
VARIANT_FILES = {
    'Baseline': "test_eval_baseline_predictions.csv",
    'Baseline + TTA': "test_eval_baseline_tta_predictions.csv",
    'Fairness-corrected': "test_eval_fairness_predictions.csv",
    'Fairness + MC-Dropout': "mc_dropout_results_test.csv",
    'Fairness + TTA': "test_eval_fairness_tta_predictions.csv",
}


def _load_predictions(variant_label):
    path = os.path.join(OUTPUTS_DIR, VARIANT_FILES[variant_label])
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _model_metrics(df):
    correct = df['true_label'] == df['predicted_label']
    overall_acc = correct.mean() * 100
    mel = df[df['true_label'] == 'MEL']
    mel_recall = (mel['predicted_label'] == 'MEL').mean() * 100 if len(mel) else float('nan')

    group_acc = {}
    for g in GROUP_ORDER:
        sub = df[df['fitzpatrick_group'] == g]
        group_acc[g] = (sub['true_label'] == sub['predicted_label']).mean() * 100 if len(sub) else float('nan')

    gap = max(group_acc.values()) - min(group_acc.values())
    return {
        'overall_acc': overall_acc, 'mel_recall': mel_recall,
        'group_acc': group_acc, 'gap': gap, 'n': len(df),
    }


def build_stat_tiles():
    """Headline KPI row for the recommended production variant (Fairness + TTA),
    with the plain baseline shown alongside each figure for context."""
    best = _load_predictions('Fairness + TTA')
    baseline = _load_predictions('Baseline')
    if best is None or baseline is None:
        return '<div class="stat-tile-row"><div class="stat-tile"><span class="stat-sub">Results not available yet.</span></div></div>'

    bm, bl = _model_metrics(best), _model_metrics(baseline)

    def tile(label, value, sub, highlight=False):
        cls = "stat-tile stat-highlight" if highlight else "stat-tile"
        return (f'<div class="{cls}"><div class="stat-label">{label}</div>'
                f'<div class="stat-value">{value}</div><div class="stat-sub">{sub}</div></div>')

    tiles = [
        tile("Recommended model", "Fairness + TTA", "best balance in ablation study", highlight=True),
        tile("Overall accuracy", f"{bm['overall_acc']:.1f}%", f"baseline: {bl['overall_acc']:.1f}%"),
        tile("Melanoma recall", f"{bm['mel_recall']:.1f}%", f"baseline: {bl['mel_recall']:.1f}% (safety-critical)"),
        tile("Fairness gap", f"{bm['gap']:.1f} pts", f"baseline: {bl['gap']:.1f} pts (lower is better)"),
    ]
    return f'<div class="stat-tile-row">{"".join(tiles)}</div>'


def build_summary_table():
    rows = []
    for label in VARIANT_FILES:
        df = _load_predictions(label)
        if df is None:
            continue
        m = _model_metrics(df)
        rows.append({
            'Model': label,
            'Overall Accuracy': f"{m['overall_acc']:.2f}%",
            'Melanoma Recall': f"{m['mel_recall']:.2f}%",
            'Light Skin (I-II)': f"{m['group_acc']['Light (I-II)']:.2f}%",
            'Medium Skin (III-IV)': f"{m['group_acc']['Medium (III-IV)']:.2f}%",
            'Dark Skin (V-VI)': f"{m['group_acc']['Dark (V-VI)']:.2f}%",
            'Fairness Gap (pts)': f"{m['gap']:.2f}",
        })
    if not rows:
        return pd.DataFrame([{"Status": "No evaluation results found yet in src/outputs/. "
                                          "Run src/models/evaluate_test.py for both models first."}])
    return pd.DataFrame(rows)


def _style_axes(ax, fig):
    fig.patch.set_facecolor(CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(GRIDLINE)
    ax.tick_params(colors=INK['secondary'], labelsize=9)
    ax.xaxis.label.set_color(INK['secondary'])
    ax.yaxis.label.set_color(INK['secondary'])
    ax.title.set_color(INK['primary'])


def build_group_accuracy_chart():
    available = [(label, _load_predictions(label)) for label in VARIANT_FILES]
    available = [(label, df) for label, df in available if df is not None]
    if not available:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Evaluation results not available yet", ha='center', va='center')
        ax.axis('off')
        return fig

    x = np.arange(len(GROUP_ORDER))
    n = len(available)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(9, 4.5))
    _style_axes(ax, fig)
    for i, (label, df) in enumerate(available):
        m = _model_metrics(df)
        vals = [m['group_acc'][g] for g in GROUP_ORDER]
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=label,
               color=CATEGORICAL[i % len(CATEGORICAL)], edgecolor=CHART_SURFACE, linewidth=1)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-skin-tone-group accuracy across all variants  (exact figures in the table above)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Light\n(I-II)', 'Medium\n(III-IV)', 'Dark\n(V-VI)'])
    ax.legend(fontsize=8, facecolor=CHART_SURFACE, edgecolor=GRIDLINE)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def build_confusion_matrix_chart(model_name='Fairness + TTA'):
    df = _load_predictions(model_name)
    if df is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Evaluation results not available yet", ha='center', va='center')
        ax.axis('off')
        return fig

    from sklearn.metrics import confusion_matrix
    from matplotlib.colors import LinearSegmentedColormap
    labels = list(CLASS_NAMES.keys())
    cm = confusion_matrix(df['true_label'], df['predicted_label'], labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    # Sequential blue ramp from the validated palette (light->dark = low->high),
    # rather than matplotlib's default 'Blues'.
    blue_ramp = LinearSegmentedColormap.from_list(
        'brand_blue', ['#cde2fb', '#3987e5', '#0d366b'])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _style_axes(ax, fig)
    im = ax.imshow(cm_norm, cmap=blue_ramp, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion matrix ({model_name}, row-normalized)')
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                         color='white' if cm_norm[i, j] > 0.5 else INK['primary'], fontsize=8)
    cbar = fig.colorbar(im, ax=ax, label='Row-normalized fraction')
    cbar.ax.yaxis.label.set_color(INK['secondary'])
    cbar.ax.tick_params(colors=INK['secondary'])
    fig.tight_layout()
    return fig


ABOUT_MD = """
# Fairness-Aware Skin Lesion Classifier

A 7-class dermatology image classifier (Melanoma, Melanocytic nevus, Basal cell
carcinoma, Actinic keratosis, Benign keratosis, Dermatofibroma, Vascular lesion)
built on EfficientNet-B4, trained on HAM10000 + ISIC2019.

This project's focus is not just accuracy, but **fairness across skin tones**.
Dermatology AI models are well known to underperform on darker skin in the literature,
largely because public dermoscopy datasets skew heavily towards lighter skin. This
project estimates each image's skin tone (via the Individual Typology Angle, since
neither source dataset has ground-truth Fitzpatrick labels) and trains a
fairness-corrected model using a focal loss with per-skin-tone-group reweighting,
alongside the plain baseline for comparison.

> **This is a research/educational prototype, not a medical device.** It has not been
> clinically validated and must not be used for actual diagnosis. Always consult a
> qualified dermatologist for any skin concern.

Use the tabs above to try the live classifier, see the fairness/accuracy comparison
between models, or read about the methodology and known limitations.
"""

METHODOLOGY_MD = """
# Methodology & Data Card

## Datasets
- **HAM10000** (10,015 images) and **ISIC2019** (25,331 images), combined and
  deduplicated down to a 7-class taxonomy (SCC and unknown-label images dropped).
- Skin tone groups (Light I-II / Medium III-IV / Dark V-VI) are *estimated*, not
  ground truth: computed from the peri-lesion border color in LAB space (Individual
  Typology Angle), since neither source dataset provides Fitzpatrick labels.

## Models
- **Baseline**: EfficientNet-B4 + BatchNorm/Linear head, plain cross-entropy loss.
- **Fairness-corrected**: same backbone, warm-started from the baseline, trained with
  focal loss (gamma=2) and inverse-frequency loss reweighting by estimated skin-tone
  group, plus a Dropout(0.3) head that also enables MC-Dropout uncertainty estimation.

## A data leakage bug we found and fixed
The original train/val/test split didn't account for **lesion grouping** -- both
source datasets photograph the same physical lesion multiple times (different angles/
zoom), and a naive split let those near-duplicate photos land in different sets. We
measured this affected **~61% of the original test set**, meaningfully inflating
reported accuracy. The split now groups all photos of a lesion together (never split
across train/val/test), which dropped the honest test accuracy from the low-80s to
the mid-70s -- a real, expected consequence of removing memorization, not a
regression.

## Known limitations
- Melanoma vs. nevus is genuinely hard to distinguish even for experts in many cases;
  melanoma recall is a persistent weak point.
- Dermatofibroma and Vascular lesion classes have very few test examples (~30-40),
  so their individual accuracy is noisy.
- The Fitzpatrick group is an algorithmic estimate (ITA), not a clinical assessment,
  and should be read as an approximation.
- Grad-CAM++ heatmaps can show a residual attention artifact in one image corner on
  low-contrast, borderless lesion photos -- a known EfficientNet architectural quirk
  (confirmed via ablation to be a fixed, content-independent activation, not the
  model attending to something real in the image).
"""


CUSTOM_THEME = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#eef5fc", c100="#cde2fb", c200="#9ec5f4", c300="#6da7ec",
        c400="#3987e5", c500="#2a78d6", c600="#256abf", c700="#184f95",
        c800="#104281", c900="#0d366b", c950="#0a2b56",
    ),
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.Font("system-ui"), gr.themes.Font("-apple-system"),
          gr.themes.Font("Segoe UI"), gr.themes.Font("sans-serif")],
).set(
    button_primary_background_fill="#2a78d6",
    button_primary_background_fill_hover="#184f95",
    button_primary_text_color="#ffffff",
)

HERO_HTML = """
<div class="hero-banner">
  <h1>&#129502; Skin Lesion Classifier</h1>
  <p>A 7-class dermatology classifier built with a specific focus on <b>fairness across
  skin tones</b> -- estimating each image's skin-tone group and correcting for the
  well-documented bias toward lighter skin in public dermoscopy datasets.</p>
  <div class="hero-badges">
    <span class="hero-badge">&#128202; EfficientNet-B4</span>
    <span class="hero-badge">&#9878; Focal loss + skin-tone reweighting</span>
    <span class="hero-badge">&#128269; Grad-CAM++ explainability</span>
    <span class="hero-badge">&#127919; MC-Dropout uncertainty</span>
  </div>
</div>
"""

EXAMPLE_IMAGES = sorted(glob.glob(os.path.join(BASE_DIR, "examples", "*.jpg")))


def build_app():
    with gr.Blocks(title="Skin Lesion Classifier - Fairness Project") as demo:
        gr.HTML(HERO_HTML)
        with gr.Tabs():
            with gr.Tab("🏠 About"):
                gr.Markdown(ABOUT_MD)

            with gr.Tab("🔬 Diagnose"):
                gr.Markdown(
                    "## Live diagnosis\n\n"
                    "Upload a **dermoscopy-style** close-up image of a skin lesion (or click "
                    "one of the samples below -- these are real held-out test images). "
                    "**Not a medical device -- for research/educational use only.**"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(type="pil", label="Lesion image")
                        gr.Examples(
                            examples=[[p] for p in EXAMPLE_IMAGES],
                            inputs=[image_input],
                            label="Or click a sample image (one per class)",
                            examples_per_page=7,
                        )
                        mode_input = gr.Radio(
                            choices=["Fast (TTA)", "Uncertainty-aware (MC-Dropout, ~30x slower)"],
                            value="Fast (TTA)",
                            label="Inference mode",
                            info="Fast (TTA) gave the best accuracy/fairness/melanoma-recall "
                                 "balance in our ablation study and is recommended by default.",
                        )
                        predict_btn = gr.Button("🔍 Diagnose", variant="primary")
                    with gr.Column(scale=1):
                        prediction_output = gr.HTML(label="Prediction")
                        uncertainty_output = gr.HTML(label="Uncertainty")
                        fitz_output = gr.HTML(label="Estimated skin tone")
                with gr.Row():
                    heatmap_output = gr.Image(label="Grad-CAM++ (what the model focused on)")
                    prob_output = gr.Plot(label="Class probabilities")

                predict_btn.click(
                    fn=predict,
                    inputs=[image_input, mode_input],
                    outputs=[heatmap_output, prob_output, prediction_output,
                             fitz_output, uncertainty_output],
                )

            with gr.Tab("📊 Fairness & Performance"):
                gr.Markdown(
                    "## Accuracy and fairness comparison across all variants\n\n"
                    "All numbers below are computed on the held-out test set, which is "
                    "lesion-disjoint from training (see Methodology tab)."
                )
                stat_tiles = gr.HTML(value=build_stat_tiles())
                summary_table = gr.Dataframe(value=build_summary_table(), label="Summary")
                with gr.Row():
                    group_chart = gr.Plot(value=build_group_accuracy_chart(),
                                           label="Per-skin-tone-group accuracy")
                with gr.Row():
                    cm_baseline = gr.Plot(value=build_confusion_matrix_chart('Baseline'),
                                           label="Baseline confusion matrix")
                    cm_fairness = gr.Plot(value=build_confusion_matrix_chart('Fairness + TTA'),
                                           label="Fairness + TTA confusion matrix")
                refresh_btn = gr.Button("🔄 Refresh from latest results")
                refresh_btn.click(
                    fn=lambda: (build_stat_tiles(), build_summary_table(), build_group_accuracy_chart(),
                                build_confusion_matrix_chart('Baseline'),
                                build_confusion_matrix_chart('Fairness + TTA')),
                    outputs=[stat_tiles, summary_table, group_chart, cm_baseline, cm_fairness],
                )

            with gr.Tab("📖 Methodology"):
                gr.Markdown(METHODOLOGY_MD)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=CUSTOM_THEME, css=CUSTOM_CSS)
