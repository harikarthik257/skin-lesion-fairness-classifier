"""
Combined baseline + fairness training for Kaggle GPU kernels.

Adapted from src/models/train_baseline.py and src/models/train_fairness.py:
- Reads images directly from Kaggle's mounted input datasets (no kagglehub download).
- Reads the leak-free, lesion-grouped manifests from the uploaded manifests dataset
  (identical split used throughout the local project -- not recomputed here).
- Trains baseline (plain cross-entropy) first, then fairness (joint skin-tone x
  disease-class weighted focal loss, warm-started from the new baseline).
- Both use patience-based early stopping (max 20 epochs, patience 5).
- Saves both best checkpoints to /kaggle/working/ for download afterward.
"""
import os
import glob
import time
import subprocess
import sys

# Kaggle's current default image ships a PyTorch build that dropped support for
# Pascal-generation GPUs (sm_60 -- e.g. the Tesla P100, which this account's free-tier
# GPU allocation is pinned to and can't be changed via the API). That build only
# supports compute capability 7.0+ (Volta and newer), so it raises "no kernel image
# available" on a P100. Reinstalling a slightly older, still-widely-used torch/
# torchvision pair (matched release, both still built with Pascal in their kernel
# set) before anything else imports torch fixes this without needing a different GPU.
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.4.1", "torchvision==0.19.1",
    "--index-url", "https://download.pytorch.org/whl/cu121",
], check=True)
# NOTE: torch==2.2.0 also supports this P100's Pascal architecture (sm_60), but it
# predates NumPy 2.0 (released June 2024) and was built against the NumPy 1.x ABI --
# downgrading numpy to match then broke every OTHER numpy>=2-compiled package already
# in Kaggle's base image (opencv, jax, cupy, etc.), a worse problem than the one it
# fixed. torch 2.4.1 (Aug 2024) postdates NumPy 2.0 and should already be built
# numpy-2-compatible, avoiding that whole conflict -- if it also still covers sm_60.

try:
    import timm  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm

print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f"GPU compute capability: sm_{major}{minor}")

# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------
INPUT_DIR = "/kaggle/input"
WORKING_DIR = "/kaggle/working"


def resolve_input_dir(keywords, label):
    """Find the actual mounted directory for a dataset by keyword match, walking
    /kaggle/input recursively -- Kaggle's mount layout varies (flat /kaggle/input/<name>
    on some environments, nested /kaggle/input/datasets/<owner>/<name> on others) and
    folder names don't always match the owner/dataset-slug used to reference it."""
    matches = []
    for root, dirs, _files in os.walk(INPUT_DIR):
        depth = root[len(INPUT_DIR):].count(os.sep)
        if depth >= 4:  # don't descend into the actual data files themselves
            dirs[:] = []
            continue
        for d in dirs:
            if any(kw.lower() in d.lower() for kw in keywords):
                matches.append(os.path.join(root, d))
    if not matches:
        all_dirs = [os.path.join(r, d) for r, dirs, _ in os.walk(INPUT_DIR) for d in dirs]
        raise FileNotFoundError(f"No directory matching {keywords} for {label} found. "
                                 f"Walked: {all_dirs[:50]}")
    # Prefer the shallowest match (closest to /kaggle/input, most likely the dataset root)
    resolved = min(matches, key=lambda p: p.count(os.sep))
    print(f"Resolved {label} -> {resolved}")
    return resolved


HAM_DIR = resolve_input_dir(["ham10000", "skin-cancer-mnist"], "HAM10000")
ISIC_DIR = resolve_input_dir(["isic"], "ISIC2019")
MANIFESTS_DIR = resolve_input_dir(["manifest", "fairness"], "manifests")

CLASS_MAPPING = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}
FITZ_MAPPING = {'Light (I-II)': 0, 'Medium (III-IV)': 1, 'Dark (V-VI)': 2}
FITZ_NAMES = {0: 'I-II', 1: 'III-IV', 2: 'V-VI'}

BATCH_SIZE = 32   # bumped up from 8 -- Kaggle GPUs have far more VRAM than the local 4GB card
NUM_WORKERS = 4


# ---------------------------------------------------------------------------
# Model definitions (mirrors src/models/model_defs.py)
# ---------------------------------------------------------------------------
class BaselineModel(nn.Module):
    def __init__(self, num_classes=7, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class FairnessModel(nn.Module):
    def __init__(self, num_classes=7, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(p=0.3),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='none'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def find_image_paths():
    print("Indexing Kaggle-mounted dataset images...")
    image_dict = {}
    for p in glob.glob(os.path.join(HAM_DIR, "**", "*.jpg"), recursive=True):
        image_dict[os.path.splitext(os.path.basename(p))[0]] = p
    for p in glob.glob(os.path.join(ISIC_DIR, "**", "*.jpg"), recursive=True):
        image_dict[os.path.splitext(os.path.basename(p))[0]] = p
    print(f"Found {len(image_dict)} unique image files.")
    return image_dict


class SkinDataset(Dataset):
    def __init__(self, csv_path, image_dict, transform=None, with_fitz=True):
        self.df = pd.read_csv(csv_path)
        self.image_dict = image_dict
        self.transform = transform
        self.with_fitz = with_fitz

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

        if with_fitz and 'fitzpatrick_group' in self.df.columns:
            self.fitz_counts = self.df['fitzpatrick_group'].value_counts().to_dict()
            fitz_idx_col = self.df['fitzpatrick_group'].map(FITZ_MAPPING)
            class_idx_col = self.df['class_label'].map(CLASS_MAPPING)
            valid = fitz_idx_col.notna() & class_idx_col.notna()
            pairs = zip(fitz_idx_col[valid].astype(int), class_idx_col[valid].astype(int))
            self.joint_counts = {}
            for g, c in pairs:
                self.joint_counts[(g, c)] = self.joint_counts.get((g, c), 0) + 1
        else:
            self.fitz_counts = {}
            self.joint_counts = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['image_id'])
        img_path = self.image_dict[img_id]

        image = Image.open(img_path).convert('RGB')
        label = CLASS_MAPPING[row['class_label']]

        if self.transform:
            image = self.transform(image)

        if self.with_fitz:
            fitz_str = row.get('fitzpatrick_group', 'unknown')
            fitz_idx = FITZ_MAPPING.get(fitz_str, -1)
            return image, label, fitz_idx
        return image, label


TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ---------------------------------------------------------------------------
# Baseline training
# ---------------------------------------------------------------------------
def train_baseline(image_dict, device):
    print("\n" + "=" * 60)
    print("BASELINE TRAINING")
    print("=" * 60)

    train_ds = SkinDataset(os.path.join(MANIFESTS_DIR, "train_manifest_fitz.csv"), image_dict,
                            transform=TRAIN_TRANSFORM, with_fitz=False)
    val_ds = SkinDataset(os.path.join(MANIFESTS_DIR, "val_manifest_fitz.csv"), image_dict,
                          transform=VAL_TRANSFORM, with_fitz=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = BaselineModel(num_classes=7, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    max_epochs, patience = 20, 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_acc, epochs_since_improvement = 0.0, 0
    best_path = os.path.join(WORKING_DIR, "baseline_efficientnet_b4_best.pt")
    total_start = time.time()

    for epoch in range(max_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        start = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1} Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")

        scheduler.step()
        train_loss /= train_total
        train_acc = 100. * train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_loss /= val_total
        val_acc = 100. * val_correct / val_total
        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{max_epochs} | Time: {elapsed:.0f}s | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_since_improvement = 0
            torch.save(model.state_dict(), best_path)
            print(f"  --> Saved new best model (Val Acc: {val_acc:.2f}%)")
        else:
            epochs_since_improvement += 1
            print(f"  No improvement for {epochs_since_improvement}/{patience} epochs.")
            if epochs_since_improvement >= patience:
                print("  Early stopping.")
                break

    print(f"\nBaseline training complete in {(time.time()-total_start)/60:.2f} min! Best Val Acc: {best_val_acc:.2f}%")
    return best_path


# ---------------------------------------------------------------------------
# Fairness training (joint skin-tone x class weighted focal loss)
# ---------------------------------------------------------------------------
def train_fairness(image_dict, device, baseline_path):
    print("\n" + "=" * 60)
    print("FAIRNESS TRAINING (joint skin-tone x class weighted focal loss)")
    print("=" * 60)

    train_ds = SkinDataset(os.path.join(MANIFESTS_DIR, "train_manifest_fitz.csv"), image_dict,
                            transform=TRAIN_TRANSFORM, with_fitz=True)
    val_ds = SkinDataset(os.path.join(MANIFESTS_DIR, "val_manifest_fitz.csv"), image_dict,
                          transform=VAL_TRANSFORM, with_fitz=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Joint (skin-tone group x disease-class) weights, floored + capped for stability
    # on tiny cells (e.g. Dark-skin Dermatofibroma may be single digits).
    MIN_CELL_COUNT, MAX_WEIGHT_RATIO = 20, 5.0
    num_groups, num_classes = len(FITZ_MAPPING), len(CLASS_MAPPING)
    joint_counts_matrix = torch.ones(num_groups, num_classes)
    for (g, c), cnt in train_ds.joint_counts.items():
        joint_counts_matrix[g, c] = cnt
    total_samples = joint_counts_matrix.sum().item()
    floored = joint_counts_matrix.clamp(min=MIN_CELL_COUNT)
    inv_freq = total_samples / floored
    joint_weights_tensor = (inv_freq / inv_freq.mean()).clamp(max=MAX_WEIGHT_RATIO).to(device)

    class_names = list(CLASS_MAPPING.keys())
    print("Joint (group x class) weights:")
    print("           " + "  ".join(f"{c:>6}" for c in class_names))
    for g, g_name in FITZ_NAMES.items():
        row = "  ".join(f"{joint_weights_tensor[g, c].item():>6.2f}" for c in range(num_classes))
        print(f"  {g_name:>8}  {row}")

    model = FairnessModel(num_classes=7).to(device)
    if os.path.exists(baseline_path):
        print(f"Loading warm start weights from {baseline_path}...")
        model.load_state_dict(torch.load(baseline_path, map_location=device), strict=False)
    else:
        print("Warning: baseline checkpoint not found! Starting from scratch.")

    criterion = FocalLoss(gamma=2.0, reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    max_epochs, patience = 20, 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_acc, epochs_since_improvement = 0.0, 0
    best_path = os.path.join(WORKING_DIR, "fairness_efficientnet_b4_best.pt")
    total_start = time.time()

    for epoch in range(max_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        start = time.time()

        for batch_idx, (inputs, targets, fitz_groups) in enumerate(train_loader):
            inputs, targets, fitz_groups = inputs.to(device), targets.to(device), fitz_groups.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss_per_sample = criterion(outputs, targets)
                sample_weights = torch.ones_like(loss_per_sample)
                valid_mask = fitz_groups >= 0
                if valid_mask.any():
                    sample_weights[valid_mask] = joint_weights_tensor[fitz_groups[valid_mask], targets[valid_mask]]
                loss = (loss_per_sample * sample_weights).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1} Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")

        scheduler.step()
        train_loss /= train_total
        train_acc = 100. * train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        fitz_correct = {0: 0, 1: 0, 2: 0}
        fitz_total = {0: 0, 1: 0, 2: 0}
        with torch.no_grad():
            for inputs, targets, fitz_groups in val_loader:
                inputs, targets, fitz_groups = inputs.to(device), targets.to(device), fitz_groups.to(device)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(inputs)
                    loss_per_sample = criterion(outputs, targets)
                    sample_weights = torch.ones_like(loss_per_sample)
                    valid_mask = fitz_groups >= 0
                    if valid_mask.any():
                        sample_weights[valid_mask] = joint_weights_tensor[fitz_groups[valid_mask], targets[valid_mask]]
                    loss = (loss_per_sample * sample_weights).mean()
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                for i in range(targets.size(0)):
                    g = fitz_groups[i].item()
                    if g in fitz_total:
                        fitz_total[g] += 1
                        if predicted[i] == targets[i]:
                            fitz_correct[g] += 1

        val_loss /= val_total
        val_acc = 100. * val_correct / val_total
        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{max_epochs} | Time: {elapsed:.0f}s | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        for g, g_name in FITZ_NAMES.items():
            g_acc = 100. * fitz_correct[g] / max(fitz_total[g], 1)
            print(f"    --> Group {g_name} Val Acc: {g_acc:.2f}% ({fitz_correct[g]}/{fitz_total[g]})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_since_improvement = 0
            torch.save(model.state_dict(), best_path)
            print(f"  --> Saved new best model (Val Acc: {val_acc:.2f}%)")
        else:
            epochs_since_improvement += 1
            print(f"  No improvement for {epochs_since_improvement}/{patience} epochs.")
            if epochs_since_improvement >= patience:
                print("  Early stopping.")
                break

    print(f"\nFairness training complete in {(time.time()-total_start)/60:.2f} min! Best Val Acc: {best_val_acc:.2f}%")
    return best_path


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    image_dict = find_image_paths()
    baseline_path = train_baseline(image_dict, device)
    fairness_path = train_fairness(image_dict, device, baseline_path)
    print(f"\nDone. Checkpoints saved to {WORKING_DIR}:")
    print(f"  {baseline_path}")
    print(f"  {fairness_path}")


if __name__ == "__main__":
    main()
