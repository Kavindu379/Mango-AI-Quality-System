import nbformat as nbf

nb = nbf.v4.new_notebook()

md1 = """# 🥭 Mango Quality AI - EfficientNet-B0 Training
This notebook trains the PyTorch EfficientNet-B0 model for the Mango Quality Assessment System.
**Requirements:**
- GPU MUST be enabled (Runtime -> Change runtime type -> Hardware accelerator: GPU).
- The dataset must be prepared (cropped by YOLO) and structured as:
  `dataset_cropped/train`, `dataset_cropped/val`, `dataset_cropped/test`.
"""

code1 = """!pip install -q torch torchvision numpy matplotlib scikit-learn pandas
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Verify GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type != 'cuda':
    print("WARNING: GPU is not enabled! Training will be very slow.")
"""

code2 = """# Configuration
EPOCHS = 35
BATCH_SIZE = 16
EARLY_STOPPING_PATIENCE = 7
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4

DATA_DIR = 'dataset_cropped'
MODEL_SAVE_PATH = 'mango_model.pth'
CLASS_NAMES = ['Grade_A_Ripe', 'Grade_B_Unripe', 'Grade_C_Overripe', 'Non_Mango']
"""

code3 = """# Dataset Definition
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
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Data Augmentation
# Hue jitter is restricted (0.02) to preserve ripeness color information!
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Loaders
train_dataset = MangoDataset(os.path.join(DATA_DIR, 'train'), transform=train_transforms)
val_dataset = MangoDataset(os.path.join(DATA_DIR, 'val'), transform=val_transforms)
test_dataset = MangoDataset(os.path.join(DATA_DIR, 'test'), transform=val_transforms)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

if len(train_dataset) == 0:
    raise ValueError(f"No training samples found in {os.path.join(DATA_DIR, 'train')}. Please verify the dataset path and structure!")

# Weighted sampling to handle class imbalance
class_counts = [0] * len(CLASS_NAMES)
for _, label in train_dataset.samples:
    class_counts[label] += 1
class_weights = [1.0 / max(c, 1) for c in class_counts]
sample_weights = [class_weights[label] for _, label in train_dataset.samples]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
"""

code4 = """# Model Architecture (EfficientNet-B0)
class MangoEfficientNetB0(nn.Module):
    def __init__(self, num_classes=4):
        super(MangoEfficientNetB0, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True), # Increased dropout to reduce overfitting
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

model = MangoEfficientNetB0(num_classes=len(CLASS_NAMES)).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Label smoothing reduces overconfidence
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
"""

code5 = """# Training Loop with Validation-based Early Stopping
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0.0
patience_counter = 0
best_epoch = 0

start_time = time.time()
print("[INFO] Starting Training...")

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    scheduler.step()
    epoch_train_loss = running_loss / total
    epoch_train_acc = (correct / total) * 100
    
    # Validate
    model.eval()
    val_running_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
            
    epoch_val_loss = val_running_loss / max(1, val_total)
    epoch_val_acc = (val_correct / max(1, val_total)) * 100
    
    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc)
    
    improved = "  <-- BEST" if epoch_val_acc > best_val_acc else ""
    print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%{improved}")
    
    # Early Stopping based strictly on Validation Accuracy
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        best_epoch = epoch
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\\n[EARLY STOP] No validation improvement for {EARLY_STOPPING_PATIENCE} epochs.")
            break

print(f"\\n[SUCCESS] Training Completed in {time.time() - start_time:.2f} seconds.")
print(f"Training Accuracy: {history['train_acc'][best_epoch-1]:.2f}%")
print(f"Validation Accuracy: {best_val_acc:.2f}% at Epoch {best_epoch}")
"""

code6 = """# Plot Training Curves
epochs_range = range(1, len(history['train_loss']) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs_range, history['train_loss'], label='Train Loss')
ax1.plot(epochs_range, history['val_loss'], label='Val Loss')
ax1.set_title('Loss')
ax1.legend()

ax2.plot(epochs_range, history['train_acc'], label='Train Acc')
ax2.plot(epochs_range, history['val_acc'], label='Val Acc')
ax2.set_title('Accuracy')
ax2.legend()
plt.show()
"""

code7 = """# Evaluate on Test Set
print("[INFO] Evaluating BEST model on TEST set...")
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = accuracy_score(all_labels, all_preds)
print(f"\\n========================================")
print(f"TEST ACCURACY: {test_acc * 100:.2f}%")
print(f"========================================\\n")

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues')
plt.title('Test Set Confusion Matrix')
plt.show()
"""

nb.cells = [
    nbf.v4.new_markdown_cell(md1),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_code_cell(code4),
    nbf.v4.new_code_cell(code5),
    nbf.v4.new_code_cell(code6),
    nbf.v4.new_code_cell(code7)
]
with open('Mango_Quality_CNN_Training.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Successfully generated Mango_Quality_CNN_Training.ipynb")
