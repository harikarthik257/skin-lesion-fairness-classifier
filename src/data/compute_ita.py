import os
import glob
import cv2
import numpy as np
import pandas as pd
import kagglehub
import random

# Configuration
HAM10000_DATASET = "kmader/skin-cancer-mnist-ham10000"
ISIC2019_DATASET = "andrewmvd/isic-2019"

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_SAMPLES_DIR = os.path.join(BASE_DIR, "outputs", "ita_samples")

os.makedirs(OUTPUT_SAMPLES_DIR, exist_ok=True)

def find_image_paths():
    print("Locating dataset images...")
    ham_path = kagglehub.dataset_download(HAM10000_DATASET)
    isic_path = kagglehub.dataset_download(ISIC2019_DATASET)

    image_dict = {}
    
    # Process HAM10000
    ham_images = glob.glob(os.path.join(ham_path, "**", "*.jpg"), recursive=True)
    for p in ham_images:
        basename = os.path.basename(p)
        name, _ = os.path.splitext(basename)
        image_dict[name] = p
        
    # Process ISIC2019
    isic_images = glob.glob(os.path.join(isic_path, "**", "*.jpg"), recursive=True)
    for p in isic_images:
        basename = os.path.basename(p)
        name, _ = os.path.splitext(basename)
        image_dict[name] = p
        
    print(f"Found {len(image_dict)} unique image files.")
    return image_dict

def extract_peri_lesion_mask(img):
    """
    Extracts the outer 15% margin of the image.
    Returns a boolean mask.
    """
    h, w = img.shape[:2]
    margin_y = int(h * 0.15)
    margin_x = int(w * 0.15)
    
    mask = np.zeros((h, w), dtype=bool)
    mask[:margin_y, :] = True
    mask[-margin_y:, :] = True
    mask[:, :margin_x] = True
    mask[:, -margin_x:] = True
    
    return mask

def compute_ita(img):
    """
    Compute ITA from BGR image.
    """
    # Convert BGR to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # OpenCV LAB ranges: L [0..255], a [0..255], b [0..255]
    # Standard LAB ranges: L [0..100], a [-128..127], b [-128..127]
    L = lab[:, :, 0].astype(np.float32) * 100.0 / 255.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    
    # Get peri-lesion mask
    margin_mask = extract_peri_lesion_mask(img)
    
    # Filter dark pixels
    dark_mask = L > 30.0
    
    valid_mask = margin_mask & dark_mask
    
    if not np.any(valid_mask):
        # Fallback to whole margin if all dark
        valid_mask = margin_mask
        
    if not np.any(valid_mask):
        # Fallback to whole image
        valid_mask = np.ones(L.shape, dtype=bool)
        
    mean_L = np.mean(L[valid_mask])
    mean_b = np.mean(b[valid_mask])
    
    ita = np.degrees(np.arctan2(mean_L - 50, mean_b))
    return ita

def assign_fitzpatrick(ita):
    if ita > 55:
        return 'Light (I-II)'
    elif 28 < ita <= 55:
        return 'Medium (III-IV)'
    else:
        return 'Dark (V-VI)'

def process_manifest(csv_name, image_dict):
    csv_path = os.path.join(PROCESSED_DIR, csv_name)
    df = pd.read_csv(csv_path)
    
    ita_scores = []
    fitz_groups = []
    image_paths_used = []
    
    print(f"Processing {csv_name} ({len(df)} images)...")
    for idx, row in df.iterrows():
        img_id = str(row['image_id'])
        if img_id not in image_dict:
            # Fallback for extensions
            if img_id.replace('.jpg', '') in image_dict:
                img_id = img_id.replace('.jpg', '')
            else:
                ita_scores.append(np.nan)
                fitz_groups.append("Unknown")
                image_paths_used.append(None)
                continue
                
        img_path = image_dict[img_id]
        image_paths_used.append(img_path)
        img = cv2.imread(img_path)
        
        if img is None:
            ita_scores.append(np.nan)
            fitz_groups.append("Unknown")
            continue
            
        ita = compute_ita(img)
        fitz = assign_fitzpatrick(ita)
        
        ita_scores.append(ita)
        fitz_groups.append(fitz)
        
        if idx % 1000 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(df)}")
            
    df['ita_score'] = ita_scores
    df['fitzpatrick_group'] = fitz_groups
    
    out_name = csv_name.replace('.csv', '_fitz.csv')
    df.to_csv(os.path.join(PROCESSED_DIR, out_name), index=False)
    
    print(f"\nDistribution for {out_name}:")
    counts = df['fitzpatrick_group'].value_counts()
    percs = df['fitzpatrick_group'].value_counts(normalize=True) * 100
    dist_df = pd.DataFrame({'Count': counts, 'Percentage': percs})
    print(dist_df)
    print("-" * 40)
    
    return df, image_paths_used

def main():
    image_dict = find_image_paths()
    
    manifests = ["train_manifest.csv", "val_manifest.csv", "test_manifest.csv"]
    
    all_processed = []
    
    for m in manifests:
        df, paths = process_manifest(m, image_dict)
        # Store for sampling
        df['__local_path'] = paths
        all_processed.append(df)
        
    combined_df = pd.concat(all_processed, ignore_index=True)
    valid_df = combined_df.dropna(subset=['__local_path'])
    
    print("\nGenerating 20 random visual samples...")
    samples = valid_df.sample(n=min(20, len(valid_df)), random_state=42)
    
    for idx, row in samples.iterrows():
        img_path = row['__local_path']
        img_id = row['image_id']
        ita = row['ita_score']
        fitz = row['fitzpatrick_group']
        
        img = cv2.imread(img_path)
        if img is not None:
            text1 = f"ITA: {ita:.2f}"
            text2 = f"Group: {fitz}"
            
            # Put text with a background rectangle for readability
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            
            (tw1, th1), _ = cv2.getTextSize(text1, font, font_scale, thickness)
            (tw2, th2), _ = cv2.getTextSize(text2, font, font_scale, thickness)
            
            cv2.rectangle(img, (5, 5), (max(tw1, tw2) + 15, 80), (0, 0, 0), -1)
            cv2.putText(img, text1, (10, 30), font, font_scale, (0, 255, 0), thickness)
            cv2.putText(img, text2, (10, 70), font, font_scale, (0, 255, 0), thickness)
            
            out_path = os.path.join(OUTPUT_SAMPLES_DIR, f"{img_id}_sample.jpg")
            cv2.imwrite(out_path, img)

    print(f"\nSamples saved to: {OUTPUT_SAMPLES_DIR}")

if __name__ == "__main__":
    main()
