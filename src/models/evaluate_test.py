import os
import glob
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import kagglehub
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from model_defs import (
    BaselineModel, FairnessModel,
    CLASS_MAPPING, INV_CLASS_MAPPING, FITZ_MAPPING, INV_FITZ_MAPPING,
)

HAM10000_DATASET = "kmader/skin-cancer-mnist-ham10000"
ISIC2019_DATASET = "andrewmvd/isic-2019"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUTS_DIR, exist_ok=True)


def find_image_paths():
    print("Locating dataset images for test evaluation...")
    ham_path = kagglehub.dataset_download(HAM10000_DATASET)
    isic_path = kagglehub.dataset_download(ISIC2019_DATASET)
    image_dict = {}
    for p in glob.glob(os.path.join(ham_path, "**", "*.jpg"), recursive=True):
        image_dict[os.path.splitext(os.path.basename(p))[0]] = p
    for p in glob.glob(os.path.join(isic_path, "**", "*.jpg"), recursive=True):
        image_dict[os.path.splitext(os.path.basename(p))[0]] = p
    print(f"Found {len(image_dict)} unique image files.")
    return image_dict


class SkinDatasetEval(Dataset):
    def __init__(self, csv_path, image_dict, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dict = image_dict
        self.transform = transform

        valid_idx = []
        for i, row in self.df.iterrows():
            img_id = str(row['image_id'])
            if img_id in self.image_dict:
                valid_idx.append(i)
            elif img_id.replace('.jpg', '') in self.image_dict:
                valid_idx.append(i)
                self.df.at[i, 'image_id'] = img_id.replace('.jpg', '')

        self.df = self.df.iloc[valid_idx].reset_index(drop=True)
        print(f"Loaded {len(self.df)} valid images from {os.path.basename(csv_path)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['image_id'])
        img_path = self.image_dict[img_id]

        image = Image.open(img_path).convert('RGB')
        label = CLASS_MAPPING[row['class_label']]

        fitz_str = row.get('fitzpatrick_group', 'unknown')
        fitz_idx = FITZ_MAPPING.get(fitz_str, -1)

        if self.transform:
            image = self.transform(image)

        return image, label, fitz_idx, img_id


MODEL_REGISTRY = {
    'baseline': (BaselineModel, "baseline_efficientnet_b4_best.pt"),
    'fairness': (FairnessModel, "fairness_efficientnet_b4_best.pt"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['baseline', 'fairness'], default='fairness',
                         help="Which trained model to evaluate on the held-out test set.")
    parser.add_argument('--tta', action='store_true',
                         help="Test-time augmentation: average predictions over the "
                              "original image plus horizontal and vertical flips.")
    parser.add_argument('--checkpoint', default=None,
                         help="Override path to a checkpoint file (defaults to the "
                              "registry entry for --model).")
    parser.add_argument('--resolution', type=int, default=380,
                         help="Input resolution to resize test images to (must match "
                              "the resolution the checkpoint was trained/fine-tuned at).")
    parser.add_argument('--tag', default=None,
                         help="Extra suffix for output filenames, to avoid colliding "
                              "with other evaluated variants of the same --model.")
    args = parser.parse_args()

    model_cls, default_ckpt_name = MODEL_REGISTRY[args.model]
    ckpt_path = args.checkpoint if args.checkpoint else os.path.join(MODELS_DIR, default_ckpt_name)

    image_dict = find_image_paths()

    test_transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_csv = os.path.join(PROCESSED_DIR, "test_manifest_fitz.csv")
    test_ds = SkinDatasetEval(test_csv, image_dict, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = model_cls(num_classes=7).to(device)
    print(f"Loading {args.model} weights from {ckpt_path}...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    tta_label = " (with TTA: orig + h-flip + v-flip)" if args.tta else ""
    print(f"Running evaluation{tta_label}...")

    results = []
    with torch.no_grad():
        for images, labels, fitz_idxs, img_ids in tqdm(test_loader, desc="Evaluating test set"):
            images = images.to(device)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                if args.tta:
                    # Flipping the already-resized/normalized tensor is equivalent to
                    # flipping the source image before transform, so this needs no
                    # extra PIL-level work -- just average softmax probs across views.
                    views = [images, torch.flip(images, dims=[-1]), torch.flip(images, dims=[-2])]
                    probs_sum = None
                    for view in views:
                        view_probs = torch.softmax(model(view).float(), dim=1)
                        probs_sum = view_probs if probs_sum is None else probs_sum + view_probs
                    probs = probs_sum / len(views)
                else:
                    outputs = model(images)
                    probs = torch.softmax(outputs.float(), dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            confidences = probs.max(dim=1).values.cpu().numpy()
            entropies = (-torch.sum(probs * torch.log(probs + 1e-9), dim=1)).cpu().numpy()
            labels_np = labels.numpy()
            fitz_np = fitz_idxs.numpy()

            for img_id, true_l, pred_l, fitz_i, conf, ent in zip(img_ids, labels_np, preds, fitz_np, confidences, entropies):
                results.append({
                    'image_id': img_id,
                    'true_label': INV_CLASS_MAPPING[int(true_l)],
                    'predicted_label': INV_CLASS_MAPPING[int(pred_l)],
                    'fitzpatrick_group': INV_FITZ_MAPPING[int(fitz_i)],
                    'confidence': float(conf),
                    'entropy': float(ent),
                })

    suffix = ("_tta" if args.tta else "") + (f"_{args.tag}" if args.tag else "")
    df = pd.DataFrame(results)
    out_csv = os.path.join(OUTPUTS_DIR, f"test_eval_{args.model}{suffix}_predictions.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved per-image predictions to {out_csv}")

    overall_acc = (df['true_label'] == df['predicted_label']).mean() * 100
    print(f"\n=== {args.model.upper()}{tta_label} MODEL — TEST SET RESULTS ({len(df)} images) ===")
    print(f"Overall Test Accuracy: {overall_acc:.2f}%")

    print("\n--- Per-class classification report ---")
    class_order = list(CLASS_MAPPING.keys())
    print(classification_report(df['true_label'], df['predicted_label'], labels=class_order, digits=4, zero_division=0))

    print("--- Confusion matrix (rows=true, cols=pred) ---")
    cm = confusion_matrix(df['true_label'], df['predicted_label'], labels=class_order)
    cm_df = pd.DataFrame(cm, index=class_order, columns=class_order)
    print(cm_df)

    print("\n--- Per-Fitzpatrick-group accuracy (fairness check on TEST set) ---")
    group_rows = []
    for group in ['Light (I-II)', 'Medium (III-IV)', 'Dark (V-VI)']:
        sub = df[df['fitzpatrick_group'] == group]
        if len(sub) == 0:
            continue
        acc = (sub['true_label'] == sub['predicted_label']).mean() * 100
        group_rows.append({'group': group, 'n': len(sub), 'accuracy': acc})
        print(f"  {group}: {acc:.2f}% ({(sub['true_label'] == sub['predicted_label']).sum()}/{len(sub)})")

    group_df = pd.DataFrame(group_rows)
    max_acc = group_df['accuracy'].max()
    min_acc = group_df['accuracy'].min()
    print(f"\n  Fairness gap (max - min group accuracy): {max_acc - min_acc:.2f} pts")

    summary_path = os.path.join(OUTPUTS_DIR, f"test_eval_{args.model}{suffix}_summary.csv")
    group_df.to_csv(summary_path, index=False)
    print(f"\nSaved group summary to {summary_path}")


if __name__ == "__main__":
    main()
