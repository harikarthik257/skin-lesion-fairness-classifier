"""
FastAPI backend for the DermLens frontend (website/derm-lens).

Wraps the same model/eval logic as the project's Gradio app (app.py) and
evaluation scripts (src/models/evaluate_test.py) behind a small JSON API,
so the React frontend can call real inference and real evaluation numbers
instead of the placeholder data it shipped with.

Run: uvicorn backend.main:app --reload --port 8000   (from the project root)
"""
import base64
import io
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "src", "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "src", "outputs")
EXAMPLES_DIR = os.path.join(BASE_DIR, "examples")

sys.path.insert(0, MODELS_DIR)
from model_defs import FairnessModel, enable_dropout, CLASS_MAPPING, INV_CLASS_MAPPING  # noqa: E402

CLASS_NAMES = {
    'MEL': 'Melanoma', 'NV': 'Melanocytic nevus', 'BCC': 'Basal cell carcinoma',
    'AKIEC': "Actinic keratosis / Bowen's disease", 'BKL': 'Benign keratosis',
    'DF': 'Dermatofibroma', 'VASC': 'Vascular lesion',
}

CLASS_DESCRIPTIONS = {
    'MEL': ("A malignant tumor of pigment-producing cells (melanocytes). The most "
            "dangerous common skin cancer because it can spread to other organs if not "
            "caught early, though it is highly treatable when detected while still "
            "confined to the skin. Often (but not always) presents as a mole that is "
            "asymmetric, has an irregular border, uneven color, or has changed recently."),
    'NV': ("A common, ordinary mole -- a benign growth of melanocytes. The large majority "
           "of moles are harmless and stable over time. This is the most frequent finding "
           "in dermoscopy datasets, which is part of why it can be visually confused with "
           "early melanoma."),
    'BCC': ("The most common form of skin cancer. It grows slowly and almost never spreads "
            "to other parts of the body, but can cause significant local tissue damage if "
            "left untreated. Usually appears on sun-exposed skin as a pearly bump, a flat "
            "scar-like area, or a sore that doesn't heal."),
    'AKIEC': ("A pre-cancerous (actinic keratosis) or early in-situ (Bowen's disease) "
              "lesion caused by cumulative sun damage. Not yet invasive cancer, but can "
              "progress to squamous cell carcinoma if untreated, so it's usually monitored "
              "or treated proactively."),
    'BKL': ("A group of benign, non-cancerous growths (including seborrheic keratosis and "
            "solar lentigo) that are extremely common, especially with age. They can look "
            "concerning -- waxy, stuck-on, or irregularly pigmented -- but carry no cancer "
            "risk themselves."),
    'DF': ("A benign, firm nodule made of fibrous tissue, often on the legs, sometimes "
           "following a minor injury like an insect bite. Harmless and typically left alone "
           "unless it's bothersome."),
    'VASC': ("A benign lesion made of blood vessels (e.g. angioma, hemangioma). Usually "
             "red, purple, or blue in color due to the blood content, and not related to "
             "pigment cells at all -- a different biological category from the other six "
             "classes."),
}

# One real held-out test image per class, used for the "try an example" flow.
EXAMPLE_FILES = {
    'MEL': 'Melanoma_ISIC_0061950.jpg',
    'NV': 'Melanocytic_Nevus_ISIC_0026084.jpg',
    'BCC': 'Basal_Cell_Carcinoma_ISIC_0067277.jpg',
    'AKIEC': 'Actinic_Keratosis_ISIC_0024925.jpg',
    'BKL': 'Benign_Keratosis_ISIC_0033531.jpg',
    'DF': 'Dermatofibroma_ISIC_0064115.jpg',
    'VASC': 'Vascular_Lesion_ISIC_0070904.jpg',
}

# Every evaluated model variant, in display order. "best" flags the current
# top single model (see PATENT_FIXES.md, Stage 1: resolution bump to 456px) --
# used to highlight it in the UI without hardcoding numbers on the frontend.
VARIANTS = [
    {"key": "baseline", "name": "Baseline", "file": "test_eval_baseline_predictions.csv", "best": False},
    {"key": "baseline_tta", "name": "Baseline + TTA", "file": "test_eval_baseline_tta_predictions.csv", "best": False},
    {"key": "fairness", "name": "Fairness-corrected", "file": "test_eval_fairness_predictions.csv", "best": False},
    {"key": "fairness_tta", "name": "Fairness + TTA", "file": "test_eval_fairness_tta_predictions.csv", "best": False},
    {"key": "fairness_tta_456", "name": "Fairness + TTA (456px)", "file": "test_eval_fairness_tta_res456_predictions.csv", "best": True},
    {"key": "fairness_mc_dropout", "name": "Fairness + MC-Dropout", "file": "mc_dropout_results_test.csv", "best": False},
]

GROUP_ORDER = ['Light (I-II)', 'Medium (III-IV)', 'Dark (V-VI)']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The live demo always runs the current best model: fairness-corrected,
# fine-tuned at 456x456 (Stage 1 of the staged accuracy-improvement plan).
INFER_RESOLUTION = 456
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "fairness_efficientnet_b4_res456_best.pt")

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((INFER_RESOLUTION, INFER_RESOLUTION)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

_MODEL = None
_CAM = None
_MODEL_LOAD_ERROR = None


def get_model():
    global _MODEL, _CAM, _MODEL_LOAD_ERROR
    if _MODEL is not None or _MODEL_LOAD_ERROR is not None:
        return _MODEL, _CAM, _MODEL_LOAD_ERROR
    try:
        model = FairnessModel(num_classes=7).to(DEVICE)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
        model.eval()
        # blocks[-1], not conv_head -- conv_head has a fixed, content-independent
        # corner-activation artifact that swamps the real Grad-CAM signal.
        cam = GradCAMPlusPlus(model=model, target_layers=[model.backbone.blocks[-1]])
        _MODEL, _CAM = model, cam
    except Exception as e:  # noqa: BLE001
        _MODEL_LOAD_ERROR = str(e)
    return _MODEL, _CAM, _MODEL_LOAD_ERROR


def estimate_fitzpatrick(pil_image):
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
        return 'Light (I-II)', float(ita)
    elif 28 < ita <= 55:
        return 'Medium (III-IV)', float(ita)
    return 'Dark (V-VI)', float(ita)


def run_inference(pil_image: Image.Image, mode: str):
    model, cam, load_error = get_model()
    if load_error is not None:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {load_error}")

    pil_image = pil_image.convert('RGB')
    resized_for_display = pil_image.resize((INFER_RESOLUTION, INFER_RESOLUTION))
    tensor = INFER_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)

    uncertainty = None
    if mode == "uncertainty":
        enable_dropout(model)
        with torch.no_grad():
            batch = tensor.repeat(30, 1, 1, 1)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                probs_all = torch.softmax(model(batch), dim=1)
        mean_probs = probs_all.mean(dim=0)
        entropy = float(-torch.sum(mean_probs * torch.log(mean_probs + 1e-9)).item())
        model.eval()
        uncertainty = {"entropy": entropy, "level": "high" if entropy > 0.8 else "low"}
    else:
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                views = [tensor, torch.flip(tensor, dims=[-1]), torch.flip(tensor, dims=[-2])]
                probs_sum = sum(torch.softmax(model(v), dim=1) for v in views)
        mean_probs = (probs_sum / len(views))[0]

    mean_probs_np = mean_probs.detach().cpu().numpy()
    pred_idx = int(mean_probs_np.argmax())
    pred_code = INV_CLASS_MAPPING[pred_idx]
    confidence = float(mean_probs_np[pred_idx] * 100)

    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0, :]
    rgb_float = np.float32(resized_for_display) / 255.0
    heatmap = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    heatmap_png = Image.fromarray(heatmap)
    buf = io.BytesIO()
    heatmap_png.save(buf, format="PNG")
    heatmap_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    fitz_group, ita = estimate_fitzpatrick(pil_image)

    return {
        "predicted_class": pred_code,
        "predicted_label": CLASS_NAMES[pred_code],
        "description": CLASS_DESCRIPTIONS[pred_code],
        "confidence": confidence,
        "class_probabilities": {
            INV_CLASS_MAPPING[i]: float(p) * 100 for i, p in enumerate(mean_probs_np)
        },
        "gradcam_heatmap": heatmap_b64,
        "estimated_skin_tone": fitz_group,
        "ita_value": ita,
        "uncertainty": uncertainty,
        "mode": mode,
    }


def compute_variant_metrics(df: pd.DataFrame):
    correct = df['true_label'] == df['predicted_label']
    overall_acc = float(correct.mean() * 100)
    mel = df[df['true_label'] == 'MEL']
    mel_recall = float((mel['predicted_label'] == 'MEL').mean() * 100) if len(mel) else None

    group_acc = {}
    for g in GROUP_ORDER:
        sub = df[df['fitzpatrick_group'] == g]
        group_acc[g] = float((sub['true_label'] == sub['predicted_label']).mean() * 100) if len(sub) else None

    valid_accs = [v for v in group_acc.values() if v is not None]
    gap = float(max(valid_accs) - min(valid_accs)) if valid_accs else None

    return {
        "overall_accuracy": overall_acc,
        "melanoma_recall": mel_recall,
        "group_accuracy": group_acc,
        "fairness_gap": gap,
        "n": int(len(df)),
    }


app = FastAPI(title="DermLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir(EXAMPLES_DIR):
    app.mount("/examples", StaticFiles(directory=EXAMPLES_DIR), name="examples")


@app.get("/api/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.get("/api/metrics")
def metrics():
    rows = []
    for variant in VARIANTS:
        path = os.path.join(OUTPUTS_DIR, variant["file"])
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        m = compute_variant_metrics(df)
        rows.append({"key": variant["key"], "name": variant["name"], "best": variant["best"], **m})
    if not rows:
        raise HTTPException(status_code=404, detail="No evaluation results found in src/outputs/.")
    return {"variants": rows, "group_order": GROUP_ORDER}


@app.get("/api/examples")
def examples():
    return {
        "examples": [
            {"code": code, "label": CLASS_NAMES[code], "image_url": f"/examples/{filename}"}
            for code, filename in EXAMPLE_FILES.items()
        ]
    }


class PredictExampleRequest(BaseModel):
    code: str
    mode: str = "fast"  # "fast" (TTA) | "uncertainty" (MC-Dropout)


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), mode: str = "fast"):
    if mode not in ("fast", "uncertainty"):
        raise HTTPException(status_code=400, detail="mode must be 'fast' or 'uncertainty'")
    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded file as an image.")
    return run_inference(pil_image, mode)


@app.post("/api/predict_example")
def predict_example(req: PredictExampleRequest):
    if req.code not in EXAMPLE_FILES:
        raise HTTPException(status_code=404, detail=f"No example for class '{req.code}'.")
    if req.mode not in ("fast", "uncertainty"):
        raise HTTPException(status_code=400, detail="mode must be 'fast' or 'uncertainty'")
    path = os.path.join(EXAMPLES_DIR, EXAMPLE_FILES[req.code])
    pil_image = Image.open(path)
    result = run_inference(pil_image, req.mode)
    result["image_url"] = f"/examples/{EXAMPLE_FILES[req.code]}"
    return result
