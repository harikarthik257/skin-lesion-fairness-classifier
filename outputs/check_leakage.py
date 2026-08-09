import pandas as pd

HAM_META = r"C:\Users\rbhar\.cache\kagglehub\datasets\kmader\skin-cancer-mnist-ham10000\versions\2\HAM10000_metadata.csv"
ISIC_META = r"C:\Users\rbhar\.cache\kagglehub\datasets\andrewmvd\isic-2019\versions\1\ISIC_2019_Training_Metadata.csv"

ham = pd.read_csv(HAM_META)[['image_id', 'lesion_id']]
isic = pd.read_csv(ISIC_META)[['image', 'lesion_id']].rename(columns={'image': 'image_id'})

# Combine lesion_id lookup tables; where an image_id appears in both, keep whichever has a non-null lesion_id
lesion_lookup = pd.concat([ham, isic], ignore_index=True)
lesion_lookup = lesion_lookup.dropna(subset=['lesion_id'])
lesion_lookup = lesion_lookup.drop_duplicates(subset='image_id')
lookup_dict = dict(zip(lesion_lookup['image_id'], lesion_lookup['lesion_id']))

print(f"Total image_id -> lesion_id mappings available: {len(lookup_dict)}")

splits = {}
for name in ['train', 'val', 'test']:
    df = pd.read_csv(f'src/data/processed/{name}_manifest.csv')
    df['lesion_id'] = df['image_id'].map(lookup_dict)
    splits[name] = df

all_df = pd.concat([df.assign(split=name) for name, df in splits.items()], ignore_index=True)
print(f"\nTotal images across all splits: {len(all_df)}")
print(f"Images with a known lesion_id: {all_df['lesion_id'].notna().sum()}")

grouped = all_df.dropna(subset=['lesion_id']).groupby('lesion_id')['split'].apply(lambda s: set(s))
leaked_lesions = grouped[grouped.apply(len) > 1]
print(f"\nLesions with images in MORE THAN ONE split (train/val/test): {len(leaked_lesions)}")

leaked_lesion_ids = set(leaked_lesions.index)
leaked_images = all_df[all_df['lesion_id'].isin(leaked_lesion_ids)]
print(f"Total images involved in leaked lesions: {len(leaked_images)}")

print("\nBreakdown of which split-pairs share lesions:")
from collections import Counter
pair_counts = Counter()
for s in leaked_lesions:
    pair_counts[tuple(sorted(s))] += 1
for pair, cnt in pair_counts.items():
    print(f"  {pair}: {cnt} lesions")

# Specifically: how many TEST images have a leaked (train-visible) lesion twin?
test_leaked = leaked_images[leaked_images['split'] == 'test']
print(f"\nTest images whose lesion also appears in train and/or val: {len(test_leaked)} "
      f"out of {len(splits['test'])} test images ({100*len(test_leaked)/len(splits['test']):.2f}%)")

val_leaked = leaked_images[leaked_images['split'] == 'val']
print(f"Val images whose lesion also appears in train and/or test: {len(val_leaked)} "
      f"out of {len(splits['val'])} val images ({100*len(val_leaked)/len(splits['val']):.2f}%)")
