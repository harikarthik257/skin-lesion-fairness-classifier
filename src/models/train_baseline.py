import os
import glob
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import kagglehub

from model_defs import BaselineModel, CLASS_MAPPING

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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['image_id'])
        img_path = self.image_dict[img_id]
        
        # Open image and ensure RGB
        image = Image.open(img_path).convert('RGB')
        label = CLASS_MAPPING[row['class_label']]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

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
    
    # Decrease num_workers if running on constrained OS
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = BaselineModel(num_classes=7, pretrained=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    epochs = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    best_val_acc = 0.0
    total_start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        start_time = time.time()
        
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
                
        val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.0f}s | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
              
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_save_path = os.path.join(MODELS_DIR, "baseline_efficientnet_b4_best.pt")
            torch.save(model.state_dict(), best_save_path)
            print(f"  --> Saved new best model to {best_save_path}")
            
        epoch_save_path = os.path.join(MODELS_DIR, f"baseline_efficientnet_b4_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), epoch_save_path)
            
    total_time = time.time() - total_start_time
    print(f"\nTraining complete in {total_time/60:.2f} minutes! Best Val Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
