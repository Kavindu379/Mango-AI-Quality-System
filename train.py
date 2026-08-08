import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from model import build_mango_cnn_model, CLASS_NAMES

EPOCHS = 15
BATCH_SIZE = 8
LEARNING_RATE = 0.0005
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

# REFINED DATA AUGMENTATION PIPELINE
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_and_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Initializing PyTorch CNN Training on device: {device}...")
    
    train_dataset = MangoDataset('dataset/train', transform=train_transforms)
    val_dataset = MangoDataset('dataset/val', transform=val_transforms)
    
    if len(train_dataset) == 0:
        print("[ERROR] No training images found in dataset/train! Please add images to dataset/train/Grade_A_Ripe, Grade_B_Unripe, Grade_C_Overripe.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) if len(val_dataset) > 0 else train_loader

    model = build_mango_cnn_model(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"[INFO] Starting PyTorch Fine-Tuning for {EPOCHS} Epochs (LR: {LEARNING_RATE})...")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        scheduler.step()
        epoch_train_loss = running_loss / total
        epoch_train_acc = (correct / total) * 100
        
        # Validation Evaluation
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
        
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%")

    training_time = time.time() - start_time
    print(f"[SUCCESS] Model Training Completed in {training_time:.2f} seconds!")
    
    # Save Model State Dict Weights
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[INFO] Saved trained model weights to '{MODEL_SAVE_PATH}'.")
    
    # Plot Training Performance Curves
    plot_performance_curves(history)
    
    # Generate Confusion Matrix
    plot_confusion_matrix(model, val_loader, device)

def plot_performance_curves(history):
    epochs_range = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(epochs_range, history['train_loss'], label='Train Loss', color='#f59e0b', linewidth=2)
    ax1.plot(epochs_range, history['val_loss'], label='Val Loss', color='#ef4444', linewidth=2, linestyle='--')
    ax1.set_title('CNN Loss Progression')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs_range, history['train_acc'], label='Train Acc', color='#10b981', linewidth=2)
    ax2.plot(epochs_range, history['val_acc'], label='Val Acc', color='#38bdf8', linewidth=2, linestyle='--')
    ax2.set_title('CNN Accuracy Progression (%)')
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
            
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Grade A', 'Grade B', 'Grade C'])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='YlOrRd', values_format='d')
    plt.title('PyTorch CNN Model Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()
    print("[INFO] Saved confusion matrix heatmap as 'confusion_matrix.png'.")

if __name__ == '__main__':
    train_and_evaluate()
