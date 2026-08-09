import os
import glob
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import kagglehub
from tqdm import tqdm

from model_defs import (
    FairnessModel, enable_dropout,
    CLASS_MAPPING, INV_CLASS_MAPPING, FITZ_MAPPING, INV_FITZ_MAPPING,
)

# Configuration
HAM10000_DATASET = "kmader/skin-cancer-mnist-ham10000"
ISIC2019_DATASET = "andrewmvd/isic-2019"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

def find_image_paths():
    print("Locating dataset images for inference...")
    ham_path = kagglehub.dataset_download(HAM10000_DATASET)
    isic_path = kagglehub.dataset_download(ISIC2019_DATASET)

    image_dict = {}
    for p in glob.glob(os.path.join(ham_path, "**", "*.jpg"), recursive=True):
        image_dict[os.path.splitext(os.path.basename(p))[0]] = p
    for p in glob.glob(os.path.join(isic_path, "**", "*.jpg"), recursive=True):
        image_dict[os.path.splitext(os.path.basename(p))[0]] = p
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
        print(f"Loaded {len(self.df)} images for MC Dropout Evaluation.")

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['val', 'test'], default='val',
                         help="Which manifest split to run MC Dropout inference on.")
    args = parser.parse_args()

    image_dict = find_image_paths()

    val_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_csv = os.path.join(PROCESSED_DIR, f"{args.split}_manifest_fitz.csv")
    val_ds = SkinDatasetEval(val_csv, image_dict, transform=val_transform)

    # We use batch_size=1 because we will duplicate each image 30 times internally
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = FairnessModel(num_classes=7).to(device)
    
    model_path = os.path.join(MODELS_DIR, "fairness_efficientnet_b4_best.pt")
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        raise FileNotFoundError(f"Model not found at {model_path}. Please train first.")
        
    # Standard evaluation mode sets BatchNorm and Dropout to eval
    model.eval()
    
    # Force Dropout layers back to train mode for MC Dropout
    enable_dropout(model)
    print("MC Dropout is ENABLED (BatchNorm is locked, Dropout is active).")
    
    num_mc_passes = 30
    results = []
    
    print(f"Starting MC Dropout Inference ({num_mc_passes} passes per image)...")
    
    # Warmup GPU
    dummy_input = torch.randn(num_mc_passes, 3, 380, 380).to(device)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            _ = model(dummy_input)
            
    for image, label, fitz_idx, img_id in tqdm(val_loader, desc="Evaluating"):
        # image is [1, 3, 380, 380]. Repeat it to [30, 3, 380, 380]
        batch_input = image.repeat(num_mc_passes, 1, 1, 1).to(device)
        
        # Benchmark latency
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                outputs = model(batch_input) # [30, 7]
                probs = F.softmax(outputs, dim=1) # [30, 7]
                
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Compute metrics
        mean_probs = probs.mean(dim=0) # [7]
        predicted_idx = mean_probs.argmax().item()
        
        # Uncertainty Metrics
        # 1. Variance: average variance across the 7 classes
        variance = probs.var(dim=0).mean().item()
        
        # 2. Entropy: information theory entropy of the mean prediction
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-9)).item()
        
        results.append({
            'image_id': img_id[0],
            'true_label': INV_CLASS_MAPPING[label.item()],
            'predicted_label': INV_CLASS_MAPPING[predicted_idx],
            'latency_ms': round(latency_ms, 2),
            'entropy': entropy,
            'variance': variance,
            'fitzpatrick_group': INV_FITZ_MAPPING[fitz_idx.item()]
        })
        
    df_results = pd.DataFrame(results)
    out_name = "mc_dropout_results.csv" if args.split == 'val' else f"mc_dropout_results_{args.split}.csv"
    out_path = os.path.join(OUTPUTS_DIR, out_name)
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved {len(df_results)} predictions to {out_path}")
    
    # Calculate some summary statistics
    acc = (df_results['true_label'] == df_results['predicted_label']).mean() * 100
    avg_latency = df_results['latency_ms'].mean()
    print(f"Overall MC Dropout Accuracy: {acc:.2f}%")
    print(f"Average Latency (30 passes): {avg_latency:.2f} ms/image")

if __name__ == "__main__":
    main()
