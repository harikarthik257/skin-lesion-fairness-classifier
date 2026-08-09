import os
import glob
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import kagglehub

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from model_defs import FairnessModel, CLASS_MAPPING

# Configurations
HAM10000_DATASET = "kmader/skin-cancer-mnist-ham10000"
ISIC2019_DATASET = "andrewmvd/isic-2019"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
GRADCAM_DIR = os.path.join(OUTPUTS_DIR, "gradcam_samples")

os.makedirs(GRADCAM_DIR, exist_ok=True)

def find_image_paths():
    ham_path = kagglehub.dataset_download(HAM10000_DATASET)
    isic_path = kagglehub.dataset_download(ISIC2019_DATASET)
    image_dict = {}
    for p in glob.glob(os.path.join(ham_path, "**", "*.jpg"), recursive=True):
        image_dict[os.path.splitext(os.path.basename(p))[0]] = p
    for p in glob.glob(os.path.join(isic_path, "**", "*.jpg"), recursive=True):
        image_dict[os.path.splitext(os.path.basename(p))[0]] = p
    return image_dict

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = FairnessModel(num_classes=7).to(device)
    model_path = os.path.join(MODELS_DIR, "fairness_efficientnet_b4_best.pt")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Configure target layer
    # NOTE: conv_head (final 1x1 channel-expansion conv before global pooling) was tried
    # first but produces a fixed, content-independent hotspot in one corner of the feature
    # map (confirmed on random-noise input: activation ~517 vs a -600..+50 range elsewhere).
    # This dominates GradCAM's per-image normalization and washes out real lesion signal.
    # blocks[-1] (the last MBConv block, still spatially meaningful) does not have this
    # issue and produces lesion-centered heatmaps instead.
    target_layers = [model.backbone.blocks[-1]]
    print(f"\nTargeting layer for Grad-CAM++: {target_layers[0]}\n")
    
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    
    # Read MC Dropout results to find a mix of correct/incorrect images
    results_csv = os.path.join(OUTPUTS_DIR, "mc_dropout_results.csv")
    if not os.path.exists(results_csv):
        raise FileNotFoundError("Please run mc_dropout_infer.py first to generate results.")
        
    df_results = pd.read_csv(results_csv)
    
    # Get 5 correct and 5 incorrect predictions
    correct_df = df_results[df_results['true_label'] == df_results['predicted_label']].sample(5, random_state=42)
    incorrect_df = df_results[df_results['true_label'] != df_results['predicted_label']].sample(5, random_state=42)
    
    sample_df = pd.concat([correct_df, incorrect_df])
    
    image_dict = find_image_paths()
    
    val_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Processing {len(sample_df)} images for Grad-CAM++...\n")
    
    for _, row in sample_df.iterrows():
        img_id = row['image_id']
        true_label = row['true_label']
        pred_label = row['predicted_label']
        status = "correct" if true_label == pred_label else "incorrect"
        
        img_path = image_dict.get(img_id)
        if not img_path:
            print(f"Warning: image path not found for {img_id}")
            continue
            
        # Load and transform image
        original_img = Image.open(img_path).convert('RGB')
        original_img_resized = original_img.resize((380, 380))
        rgb_img = np.float32(original_img_resized) / 255.0
        
        input_tensor = val_transform(original_img).unsqueeze(0).to(device)
        
        # Target the actual predicted class to see what drove that decision
        pred_idx = CLASS_MAPPING[pred_label]
        targets = [ClassifierOutputTarget(pred_idx)]
        
        # Generate CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        # Overlay
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_img_resized)
        axes[0].set_title(f"Original Image\nTrue: {true_label}")
        axes[0].axis('off')
        
        axes[1].imshow(visualization)
        axes[1].set_title(f"Grad-CAM++\nPredicted: {pred_label} ({status})")
        axes[1].axis('off')
        
        save_path = os.path.join(GRADCAM_DIR, f"{img_id}_{status}.png")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved: {save_path}")

    print(f"\nAll Grad-CAM++ visualizations saved to {GRADCAM_DIR}")

if __name__ == "__main__":
    main()
