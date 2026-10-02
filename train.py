import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from model import build_mango_cnn_model, CLASS_NAMES

EPOCHS = 40              # Increased from 15 — more convergence time
BATCH_SIZE = 16          # Increased from 8 — more stable gradient estimates
LEARNING_RATE = 0.001   # Slightly higher initial LR with cosine annealing
EARLY_STOPPING_PATIENCE = 10  # Stop if no val improvement for 10 epochs
MODEL_SAVE_PATH = 'mango_model.pth'

class MangoDataset(Dataset):
    """PyTorch Dataset loader supporting custom mobile photo directories."""
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

# ENHANCED DATA AUGMENTATION PIPELINE (Simulates real-world camera & lighting conditions)
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),                                    # Slightly larger before crop
    transforms.RandomCrop((224, 224)),                               # Random crop = position invariance
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),                           # NEW: vertical flip
    transforms.RandomRotation(degrees=45),                           # Increased from 20 to 45
    transforms.ColorJitter(brightness=0.4, contrast=0.4,            # Stronger color shifts
                           saturation=0.5, hue=0.15),               # NEW: hue shift (green->yellow transitions)
    transforms.RandomGrayscale(p=0.05),                             # NEW: occasional grayscale robustness
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),        # NEW: slight position shift
                            scale=(0.85, 1.15)),                     # NEW: zoom in/out variation
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),     # NEW: camera angle variation
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),      # NEW: motion/focus blur
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def mixup_data(x, y, alpha=0.4, device='cpu'):
    """
    Mixup Augmentation: Blends two training images together.
    Forces smoother decision boundaries between Grade A / B / C.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Computes Mixup loss as weighted combination of two class losses."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_and_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Initializing EfficientNet-B0 Transfer Learning Training on device: {device}...")
    
    train_dataset = MangoDataset('dataset/train', transform=train_transforms)
    val_dataset = MangoDataset('dataset/val', transform=val_transforms)
    
    if len(train_dataset) == 0:
        print("[ERROR] No training images found in dataset/train! Please add images to dataset/train/Grade_A_Ripe, Grade_B_Unripe, Grade_C_Overripe, Non_Mango.")
        return

    # WeightedRandomSampler: Ensures each class gets equal representation per batch
    # Prevents the model from being biased toward the majority class
    class_counts = [0] * len(CLASS_NAMES)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    class_weights = [1.0 / max(c, 1) for c in class_counts]
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    print(f"[INFO] Class distribution: { {CLASS_NAMES[i]: class_counts[i] for i in range(len(CLASS_NAMES))} }")
    print(f"[INFO] WeightedRandomSampler enabled for balanced batch sampling.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) if len(val_dataset) > 0 else train_loader

    # Use EfficientNet-B0 Transfer Learning by default
    model = build_mango_cnn_model(num_classes=len(CLASS_NAMES), model_type='efficientnet').to(device)
    criterion = FocalLoss(gamma=2.0)
    
    # STAGE 1 OPTIMIZER: Train Classifier Head Only
    stage1_epochs = 5
    optimizer = optim.Adam(model.get_classifier_params(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage1_epochs)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Early Stopping & Best Model Tracking
    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0

    print(f"[INFO] STAGE 1: Training Classifier Head for {stage1_epochs} Epochs (Frozen Backbone, LR: {LEARNING_RATE})...")
    print(f"[INFO] Early Stopping: Patience = {EARLY_STOPPING_PATIENCE} epochs. Best model auto-saved.")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Transition to Stage 2: Unfreeze Backbone & Use Differential Learning Rates
        if epoch == stage1_epochs + 1:
            print("\n[INFO] TRANSITIONING TO STAGE 2: Unfreezing Upper EfficientNet-B0 Backbone for Fine-Tuning!")
            if hasattr(model, 'unfreeze_backbone'):
                model.unfreeze_backbone(unfreeze_from_block=5)
                backbone_params = model.get_backbone_params()
                head_params = model.get_classifier_params()
                optimizer = optim.Adam([
                    {'params': backbone_params, 'lr': 1e-5},  # Tiny LR for pre-trained feature weights
                    {'params': head_params, 'lr': 1e-4}      # Standard LR for classifier head
                ], weight_decay=1e-4)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - stage1_epochs)
                print("[INFO] Configured Differential Learning Rates: Backbone LR = 1e-5, Head LR = 1e-4\n")

        # --- TRAINING LOOP ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Use Mixup only in Stage 2 (backbone unfrozen) for better boundary learning
        use_mixup = (epoch > stage1_epochs)

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_mixup and np.random.random() < 0.5:  # Apply Mixup 50% of batches in Stage 2
                mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.4, device=device)
                outputs = model(mixed_images)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                # For accuracy tracking use original labels
                _, predicted = torch.max(outputs, 1)
                correct += (lam * (predicted == labels_a).float() + (1 - lam) * (predicted == labels_b).float()).sum().item()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

        scheduler.step()
        epoch_train_loss = running_loss / total
        epoch_train_acc = (correct / total) * 100

        # --- VALIDATION LOOP ---
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_running_loss / max(1, val_total)
        epoch_val_acc = (val_correct / max(1, val_total)) * 100

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        stage_label = "Stage 1" if epoch <= stage1_epochs else "Stage 2 (Fine-Tuning)"
        improved = "  <-- BEST" if epoch_val_acc > best_val_acc else ""
        print(f"[{stage_label}] Epoch [{epoch:02d}/{EPOCHS:02d}] | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%{improved}")

        # --- EARLY STOPPING & BEST MODEL SAVING ---
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)  # Save BEST model, not last epoch
            print(f"   [SAVED] New best model saved! Val Acc: {best_val_acc:.2f}% at epoch {best_epoch}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n[EARLY STOP] No improvement for {EARLY_STOPPING_PATIENCE} epochs. Best Val Acc: {best_val_acc:.2f}% at epoch {best_epoch}.")
                break

    training_time = time.time() - start_time
    print(f"\n[SUCCESS] EfficientNet-B0 2-Stage Fine-Tuning Completed in {training_time:.2f} seconds!")
    print(f"[SUCCESS] Best Validation Accuracy: {best_val_acc:.2f}% at Epoch {best_epoch} (saved to '{MODEL_SAVE_PATH}')")

    # Reload best saved model for confusion matrix evaluation
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    print(f"[INFO] Reloaded best model weights for final evaluation.")

    # Plot Training Performance Curves
    plot_performance_curves(history)

    # Generate Confusion Matrix on Best Model
    plot_confusion_matrix(model, val_loader, device)

def plot_performance_curves(history):
    epochs_range = range(1, len(history['train_loss']) + 1)  # Use actual epochs run (early stop may reduce this)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(epochs_range, history['train_loss'], label='Train Loss', color='#f59e0b', linewidth=2)
    ax1.plot(epochs_range, history['val_loss'], label='Val Loss', color='#ef4444', linewidth=2, linestyle='--')
    ax1.set_title('EfficientNet-B0 Loss Progression')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs_range, history['train_acc'], label='Train Acc', color='#10b981', linewidth=2)
    ax2.plot(epochs_range, history['val_acc'], label='Val Acc', color='#38bdf8', linewidth=2, linestyle='--')
    ax2.set_title('EfficientNet-B0 Accuracy Progression (%)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_performance.png', dpi=300)
    plt.close()
    print("[INFO] Saved training loss & accuracy plot as 'training_performance.png'.")

def plot_confusion_matrix(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Grade A', 'Grade B', 'Grade C', 'Non Mango'])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='YlOrRd', values_format='d')
    plt.title('EfficientNet-B0 Model Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()
    print("[INFO] Saved confusion matrix heatmap as 'confusion_matrix.png'.")

if __name__ == '__main__':
    train_and_evaluate()
