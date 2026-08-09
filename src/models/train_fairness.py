import os
import glob
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import kagglehub

from model_defs import FairnessModel, CLASS_MAPPING, FITZ_MAPPING, FITZ_NAMES

# Configuration
HAM10000_DATASET = "kmader/skin-cancer-mnist-ham10000"
ISIC2019_DATASET = "andrewmvd/isic-2019"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

def find_image_paths():
    print("Locating dataset images for training...")
    ham_path = kagglehub.dataset_download(HAM10000_DATASET)
    isic_path = kagglehub.dataset_download(ISIC2019_DATASET)

    image_dict = {}
    
    ham_images = glob.glob(os.path.join(ham_path, "**", "*.jpg"), recursive=True)
    for p in ham_images:
        name = os.path.splitext(os.path.basename(p))[0]
        image_dict[name] = p
        
    isic_images = glob.glob(os.path.join(isic_path, "**", "*.jpg"), recursive=True)
    for p in isic_images:
        name = os.path.splitext(os.path.basename(p))[0]
        image_dict[name] = p
        
    print(f"Found {len(image_dict)} unique image files.")
    return image_dict

class SkinDataset(Dataset):
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
        
        # Calculate fitzpatrick group frequencies
        if 'fitzpatrick_group' in self.df.columns:
            self.fitz_counts = self.df['fitzpatrick_group'].value_counts().to_dict()
        else:
            self.fitz_counts = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['image_id'])
        img_path = self.image_dict[img_id]
        
        # Open image and ensure RGB
        image = Image.open(img_path).convert('RGB')
        label = CLASS_MAPPING[row['class_label']]
        
        fitz_str = row.get('fitzpatrick_group', 'unknown')
        fitz_idx = FITZ_MAPPING.get(fitz_str, -1)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, fitz_idx

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

def main():
    image_dict = find_image_paths()
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_csv = os.path.join(PROCESSED_DIR, "train_manifest_fitz.csv")
    val_csv = os.path.join(PROCESSED_DIR, "val_manifest_fitz.csv")
    
    train_ds = SkinDataset(train_csv, image_dict, transform=train_transform)
    val_ds = SkinDataset(val_csv, image_dict, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Compute Fitzpatrick group weights
    counts = [train_ds.fitz_counts.get(k, 1) for k in ['Light (I-II)', 'Medium (III-IV)', 'Dark (V-VI)']]
    total = sum(counts)
    inv_freq = [total / max(c, 1) for c in counts]
    mean_inv = sum(inv_freq) / len(inv_freq)
    fitz_weights_tensor = torch.tensor([f / mean_inv for f in inv_freq], dtype=torch.float32).to(device)
    
    print("Fitzpatrick Group Weights (I-II, III-IV, V-VI):", fitz_weights_tensor.tolist())
    
    model = FairnessModel(num_classes=7).to(device)

    # Clean, single continuous run -- warm-start from the plain (leak-free) baseline.
    # A prior attempt paused this training after epoch 1 and resumed with a reset
    # optimizer/LR schedule, which confounded the result (couldn't tell if a worse
    # outcome was the method or the resume). This run avoids that entirely.
    baseline_path = os.path.join(MODELS_DIR, "baseline_efficientnet_b4_best.pt")
    if os.path.exists(baseline_path):
        print(f"Loading warm start weights from {baseline_path}...")
        # Strict=False because we added a Dropout layer to the head which doesn't have weights,
        # but the keys will match anyway since Dropout is stateless.
        model.load_state_dict(torch.load(baseline_path, map_location=device), strict=False)
    else:
        print("Warning: Baseline model not found! Starting from scratch.")

    criterion = FocalLoss(gamma=2.0, reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    max_epochs = 20
    patience = 5  # stop if val acc doesn't improve for this many consecutive epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_acc = 0.0
    epochs_since_improvement = 0
    total_start_time = time.time()

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        start_time = time.time()
        
        for batch_idx, (inputs, targets, fitz_groups) in enumerate(train_loader):
            inputs, targets, fitz_groups = inputs.to(device), targets.to(device), fitz_groups.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                
                # Compute Focal Loss per sample
                loss_per_sample = criterion(outputs, targets)
                
                # Apply per-group weights
                sample_weights = torch.ones_like(loss_per_sample)
                valid_mask = fitz_groups >= 0
                if valid_mask.any():
                    sample_weights[valid_mask] = fitz_weights_tensor[fitz_groups[valid_mask]]
                
                # Final mean loss for the batch
                loss = (loss_per_sample * sample_weights).mean()
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 100 == 0:
                mem_alloc = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
                mem_res = torch.cuda.memory_reserved() / (1024 * 1024) if torch.cuda.is_available() else 0.0
                print(f"  Epoch {epoch+1} Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f} | GPU Mem: {mem_alloc:.1f}MB alloc, {mem_res:.1f}MB reserved")
            
        scheduler.step()
        
        train_loss = train_loss / train_total
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
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
                        sample_weights[valid_mask] = fitz_weights_tensor[fitz_groups[valid_mask]]
                    
                    loss = (loss_per_sample * sample_weights).mean()
                    
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # Per-group tracking
                for i in range(targets.size(0)):
                    g = fitz_groups[i].item()
                    if g in fitz_total:
                        fitz_total[g] += 1
                        if predicted[i] == targets[i]:
                            fitz_correct[g] += 1
                
        val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total
        
        elapsed = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{max_epochs} | Time: {elapsed:.0f}s | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        for g, g_name in FITZ_NAMES.items():
            g_acc = 100. * fitz_correct[g] / max(fitz_total[g], 1)
            print(f"    --> Group {g_name} Val Acc: {g_acc:.2f}% ({fitz_correct[g]}/{fitz_total[g]})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_since_improvement = 0
            best_save_path = os.path.join(MODELS_DIR, "fairness_efficientnet_b4_best.pt")
            torch.save(model.state_dict(), best_save_path)
            print(f"  [!] Saved new best model to {best_save_path}")
        else:
            epochs_since_improvement += 1
            print(f"  No improvement for {epochs_since_improvement}/{patience} epochs.")
            if epochs_since_improvement >= patience:
                print(f"  Early stopping: no improvement in {patience} consecutive epochs.")
                break

    total_time = time.time() - total_start_time
    print(f"\nFairness training complete in {total_time/60:.2f} minutes! Best Val Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
