import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import pandas as pd
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

CLASS_NAMES = ['Grade_A_Ripe', 'Grade_B_Unripe', 'Grade_C_Overripe', 'Non_Mango']
DATA_DIR = 'dataset_cropped/test'

class MangoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for class_index, class_name in enumerate(CLASS_NAMES):
            class_path = os.path.join(root_dir, class_name)
            if os.path.exists(class_path):
                for filename in os.listdir(class_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        img_path = os.path.join(class_path, filename)
                        self.samples.append((img_path, class_index))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, label

def build_model(device):
    # Match the architecture exactly
    model = models.efficientnet_b0()
    num_ftrs = model.classifier[1].in_features
    # We must match the saved model state dict structure.
    # New notebook uses Dropout(p=0.5). We'll set p=0.0 for eval anyway.
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(num_ftrs, len(CLASS_NAMES))
    )
    return model.to(device)

def build_old_model(device):
    # In model.py, dropout was probably default (0.2)
    model = models.efficientnet_b0()
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_ftrs, len(CLASS_NAMES))
    )
    return model.to(device)

def evaluate_model(model, loader, device, model_name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    p, r, f, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    print(f"\\n{'='*50}")
    print(f"MODEL: {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1 Score:  {f:.4f}\\n")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    return acc, p, r, f

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Test directory '{DATA_DIR}' not found.")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = MangoDataset(DATA_DIR, transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    old_model_path = 'mango_model_old.pth'
    new_model_path = 'mango_model.pth'
    
    # 1. Evaluate Old Model
    if os.path.exists(old_model_path):
        old_model = build_old_model(device)
        try:
            old_model.load_state_dict(torch.load(old_model_path, map_location=device), strict=False)
            evaluate_model(old_model, test_loader, device, "A. EXISTING OLD MODEL (Trained on Uncropped Images)")
        except Exception as e:
            print(f"Error loading old model: {e}")
    else:
        print(f"Please rename your current model to '{old_model_path}' before running this script.")
        
    # 2. Evaluate New Model
    if os.path.exists(new_model_path):
        new_model = build_model(device)
        try:
            new_model.load_state_dict(torch.load(new_model_path, map_location=device), strict=False)
            evaluate_model(new_model, test_loader, device, "B. NEW RETRAINED MODEL (Trained on Cropped Images)")
        except Exception as e:
            print(f"Error loading new model: {e}")
    else:
        print(f"New model '{new_model_path}' not found. Please train in Colab and download it here.")

if __name__ == '__main__':
    main()
