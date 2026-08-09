import os
import glob
import numpy as np
import pandas as pd
import kagglehub

# Configuration
# Kaggle dataset identifiers
HAM10000_DATASET = "kmader/skin-cancer-mnist-ham10000"
ISIC2019_DATASET = "andrewmvd/isic-2019" # You may need to change this if another Kaggle ISIC-2019 repo is preferred

# Base directory for outputs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Ensure output directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Define 7-class dictionary (matching the 7 skin disease classes objective)
# ISIC 2019 includes an 8th class: Squamous Cell Carcinoma (SCC). 
# As per the requirements, we drop SCC to maintain the 7-class structure.
CLASS_MAPPING = {
    'MEL': 0,    # Melanoma
    'NV': 1,     # Melanocytic nevus
    'BCC': 2,    # Basal cell carcinoma
    'AKIEC': 3,  # Actinic keratosis / Bowen's disease
    'BKL': 4,    # Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)
    'DF': 5,     # Dermatofibroma
    'VASC': 6    # Vascular lesion
}

def download_and_load_datasets():
    print(f"Downloading HAM10000 from Kaggle: {HAM10000_DATASET}")
    ham_path = kagglehub.dataset_download(HAM10000_DATASET)
    print(f"HAM10000 downloaded to: {ham_path}")

    print(f"Downloading ISIC 2019 from Kaggle: {ISIC2019_DATASET}")
    isic_path = kagglehub.dataset_download(ISIC2019_DATASET)
    print(f"ISIC 2019 downloaded to: {isic_path}")

    # Discover CSV files in downloaded directories
    ham_csv_path = glob.glob(os.path.join(ham_path, "**", "*.csv"), recursive=True)
    isic_csv_path = glob.glob(os.path.join(isic_path, "**", "*GroundTruth*.csv"), recursive=True) 
    if not isic_csv_path:
        isic_csv_path = glob.glob(os.path.join(isic_path, "**", "*.csv"), recursive=True)

    ham_csv = [p for p in ham_csv_path if 'metadata' in p.lower() or 'ham10000' in p.lower()][0]
    isic_csv = [p for p in isic_csv_path if 'metadata' in p.lower() or 'groundtruth' in p.lower() or 'labels' in p.lower()][0]

    # ISIC2019 also ships a separate Metadata csv (distinct from GroundTruth) which carries
    # lesion_id groupings needed to prevent lesion-level leakage across splits.
    isic_all_csv = glob.glob(os.path.join(isic_path, "**", "*.csv"), recursive=True)
    isic_meta_matches = [p for p in isic_all_csv if 'metadata' in p.lower()]
    isic_meta_csv = isic_meta_matches[0] if isic_meta_matches else None

    # Load dataframes
    df_ham = pd.read_csv(ham_csv)
    df_isic = pd.read_csv(isic_csv)
    df_isic_meta = pd.read_csv(isic_meta_csv) if isic_meta_csv else None

    print(f"Loaded HAM10000 ({len(df_ham)} rows)")
    print(f"Loaded ISIC2019 ({len(df_isic)} rows)")
    if df_isic_meta is not None:
        print(f"Loaded ISIC2019 Metadata ({len(df_isic_meta)} rows) for lesion_id grouping")
    else:
        print("Warning: ISIC2019 Metadata csv (with lesion_id) not found!")

    return df_ham, df_isic, df_isic_meta, ham_path, isic_path

def process_datasets(df_ham, df_isic, df_isic_meta, ham_dir, isic_dir):
    # Standardize column names (assuming standard Kaggle formatting)
    # HAM10000 columns often include: image_id, dx
    # ISIC2019 columns often include: image, MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK 
    # Or just image, diagnosis

    # Let's standardize them to: image_id, class_label, dataset_source, image_path

    # Process HAM10000
    df_ham_std = pd.DataFrame()
    df_ham_std['image_id'] = df_ham['image_id']
    df_ham_std['class_label'] = df_ham['dx'].str.upper()
    df_ham_std['dataset_source'] = 'HAM10000'
    # HAM10000 explicitly groups multiple photos of the same physical lesion under one
    # lesion_id (~1956 lesions have >1 image). This MUST be preserved so a lesion never
    # ends up split across train/val/test, or the model can partially memorize it.
    df_ham_std['lesion_id'] = df_ham['lesion_id'] if 'lesion_id' in df_ham.columns else pd.NA

    # Process ISIC2019
    df_isic_std = pd.DataFrame()
    if 'image' in df_isic.columns:
        df_isic_std['image_id'] = df_isic['image']
    elif 'image_id' in df_isic.columns:
        df_isic_std['image_id'] = df_isic['image_id']
    else:
        df_isic_std['image_id'] = df_isic.iloc[:, 0] # assume first col is ID

    # ISIC2019 usually has one-hot encoded labels, we convert them to categorical
    class_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    present_cols = [c for c in class_cols if c in df_isic.columns]
    
    if present_cols:
        df_isic_std['class_label'] = df_isic[present_cols].idxmax(axis=1).str.upper()
        # AK in ISIC2019 is equivalent to AKIEC in HAM10000
        df_isic_std['class_label'] = df_isic_std['class_label'].replace({'AK': 'AKIEC'})
    elif 'dx' in df_isic.columns:
        df_isic_std['class_label'] = df_isic['dx'].str.upper()
    elif 'diagnosis' in df_isic.columns:
        df_isic_std['class_label'] = df_isic['diagnosis'].str.upper()

    df_isic_std['dataset_source'] = 'ISIC2019'

    # ISIC2019's lesion_id grouping lives in a separate Metadata csv (not GroundTruth).
    # Many ISIC2019 images have no lesion_id at all (they're the only photo of that lesion).
    if df_isic_meta is not None:
        meta_id_col = 'image' if 'image' in df_isic_meta.columns else 'image_id'
        isic_lesion_lookup = df_isic_meta[[meta_id_col, 'lesion_id']].rename(columns={meta_id_col: 'image_id'})
        df_isic_std = df_isic_std.merge(isic_lesion_lookup, on='image_id', how='left')
    else:
        df_isic_std['lesion_id'] = pd.NA

    # Combine
    df_combined = pd.concat([df_ham_std, df_isic_std], ignore_index=True)

    # Filter out SCC and UNK to match the 7 classes exactly
    print(f"Total rows before dropping SCC/UNK: {len(df_combined)}")
    df_combined = df_combined[df_combined['class_label'].isin(CLASS_MAPPING.keys())]
    print(f"Total rows after filtering to 7 classes: {len(df_combined)}")

    # Deduplication
    # ISIC2019 includes HAM10000 as a subset. Match by image_id.
    # Where an ID exists in both, keep only the ISIC2019 version and drop the HAM10000 duplicate.
    print("Deduplicating by image_id...")
    df_combined = df_combined.sort_values(by='dataset_source', ascending=False)
    # Sorting 'ISIC2019' before 'HAM10000' ensures ISIC2019 is kept when dropping duplicates
    df_combined = df_combined.drop_duplicates(subset='image_id', keep='first')

    # Images with no known lesion_id (not grouped with any other photo) get a synthetic
    # group of their own -- they're still eligible for group-based splitting, just as a
    # singleton group, so every row has a usable group key.
    no_group = df_combined['lesion_id'].isna()
    df_combined.loc[no_group, 'lesion_id'] = 'SOLO_' + df_combined.loc[no_group, 'image_id'].astype(str)
    print(f"Images with a real (multi-photo) lesion_id group: {(~no_group).sum()}")
    print(f"Images with no known grouping (singleton lesion groups): {no_group.sum()}")

    print(f"Total rows after deduplication: {len(df_combined)}")

    return df_combined

def split_and_save(df, seed=42):
    """
    Group-aware, class-stratified 70/15/15 split.

    A plain stratified split (train_test_split on individual rows) ignores that
    HAM10000/ISIC2019 photograph the same physical lesion multiple times under a shared
    lesion_id. Splitting by row lets near-duplicate photos of one lesion land in both
    train and test, so the model can partially memorize test lesions -- this was measured
    to affect ~61% of a naive test split here. Instead we assign whole lesion_id groups
    (never splitting a group across sets), allocated per-class so class balance is still
    approximately preserved.
    """
    rng = np.random.RandomState(seed)

    train_parts, val_parts, test_parts = [], [], []

    for class_label, class_df in df.groupby('class_label'):
        # Each lesion_id group must be assigned to exactly one split, as a whole.
        groups = class_df.groupby('lesion_id')
        group_ids = list(groups.groups.keys())
        rng.shuffle(group_ids)

        class_total = len(class_df)
        train_target = 0.70 * class_total
        val_target = 0.85 * class_total  # cumulative through val

        train_ids, val_ids, test_ids = [], [], []
        running = 0
        for gid in group_ids:
            gsize = len(groups.get_group(gid))
            if running < train_target:
                train_ids.append(gid)
            elif running < val_target:
                val_ids.append(gid)
            else:
                test_ids.append(gid)
            running += gsize

        train_parts.append(class_df[class_df['lesion_id'].isin(train_ids)])
        val_parts.append(class_df[class_df['lesion_id'].isin(val_ids)])
        test_parts.append(class_df[class_df['lesion_id'].isin(test_ids)])

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    # Sanity check: no lesion_id should appear in more than one split.
    train_groups = set(train_df['lesion_id'])
    val_groups = set(val_df['lesion_id'])
    test_groups = set(test_df['lesion_id'])
    overlap = (train_groups & val_groups) | (train_groups & test_groups) | (val_groups & test_groups)
    if overlap:
        raise RuntimeError(f"Lesion-group leakage detected across splits: {len(overlap)} lesion_ids overlap!")
    print(f"\nLeakage check passed: 0 lesion_id groups shared across train/val/test "
          f"({len(train_groups)}/{len(val_groups)}/{len(test_groups)} unique groups per split).")

    # Save to CSV
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train_manifest.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val_manifest.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test_manifest.csv"), index=False)

    print("\n--- Split Distributions ---")
    print(f"Train set ({len(train_df)} rows):")
    print(train_df['class_label'].value_counts())

    print(f"\nValidation set ({len(val_df)} rows):")
    print(val_df['class_label'].value_counts())

    print(f"\nTest set ({len(test_df)} rows):")
    print(test_df['class_label'].value_counts())

if __name__ == "__main__":
    df_ham_raw, df_isic_raw, df_isic_meta_raw, ham_dir, isic_dir = download_and_load_datasets()
    df_clean = process_datasets(df_ham_raw, df_isic_raw, df_isic_meta_raw, ham_dir, isic_dir)
    split_and_save(df_clean)
    print("\nData preparation complete! Manifests saved to data/processed/")
