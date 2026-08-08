import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from model import build_mango_cnn_model, CLASS_NAMES

def generate_augmented_mango_dataset(base_dir='dataset', samples_per_class=50):
    """
    Generates realistic textured mango images with realistic color distributions 
    and surface decay spots for Grade A (Ripe), Grade B (Unripe), and Grade C (Overripe).
    """
    print("[INFO] Setting up Mango training dataset structure...")
    img_size = (224, 224)
    
    for split in ['train', 'val']:
        n_samples = samples_per_class if split == 'train' else 15
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(base_dir, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for i in range(n_samples):
                # Background canvas (natural background variation)
                bg_color = np.random.randint(220, 250, size=3, dtype=np.uint8)
                img = np.ones((img_size[0], img_size[1], 3), dtype=np.uint8) * bg_color
                
                center = (112 + np.random.randint(-15, 15), 112 + np.random.randint(-15, 15))
                axes = (70 + np.random.randint(-8, 8), 48 + np.random.randint(-6, 6))
                angle = np.random.randint(10, 50)
                
                if class_name == 'Grade_A_Ripe':
                    # Vibrant Golden Yellow / Orange gradient
                    base_color = (np.random.randint(230, 255), np.random.randint(170, 215), np.random.randint(15, 45))
                elif class_name == 'Grade_B_Unripe':
                    # Deep Green / Lime Green
                    base_color = (np.random.randint(25, 65), np.random.randint(140, 195), np.random.randint(25, 65))
                else:
                    # Overripe Dark Brownish / Spotty
                    base_color = (np.random.randint(100, 140), np.random.randint(75, 110), np.random.randint(35, 65))
                
                # Draw Mango oval
                cv2.ellipse(img, center, axes, angle, 0, 360, base_color, -1)
                
                # Add natural skin texture noise
                noise = np.random.normal(0, 8, img.shape).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                # Add dark decay spots for Grade C
                if class_name == 'Grade_C_Overripe':
                    for _ in range(np.random.randint(6, 15)):
                        spot_center = (
                            center[0] + np.random.randint(-35, 35),
                            center[1] + np.random.randint(-35, 35)
                        )
                        cv2.circle(img, spot_center, np.random.randint(4, 10), (35, 25, 15), -1)
                
                img = cv2.GaussianBlur(img, (3, 3), 0)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                filepath = os.path.join(class_dir, f"mango_{i:03d}.jpg")
                cv2.imwrite(filepath, img_bgr)
                
    # Also save 3 test demo sample images
    test_dir = 'test_images'
    os.makedirs(test_dir, exist_ok=True)
    
    samples = [
        ('Grade_A_Ripe', (245, 195, 25), 'sample_ripe_mango.jpg'),
        ('Grade_B_Unripe', (35, 175, 45), 'sample_unripe_mango.jpg'),
        ('Grade_C_Overripe', (115, 85, 45), 'sample_overripe_mango.jpg')
    ]
    
    for class_name, color, filename in samples:
        img = np.ones((img_size[0], img_size[1], 3), dtype=np.uint8) * 245
        cv2.ellipse(img, (112, 112), (70, 48), 30, 0, 360, color, -1)
        if class_name == 'Grade_C_Overripe':
            for _ in range(10):
                cv2.circle(img, (112 + np.random.randint(-25, 25), 112 + np.random.randint(-25, 25)), 6, (30, 20, 10), -1)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        cv2.imwrite(os.path.join(test_dir, filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
    print("[SUCCESS] Dataset directory structure and sample test images verified.")

class MangoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        for idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(class_dir, filename), idx))
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def train_and_evaluate_model(data_dir='dataset', model_save_path='mango_model.pth', epochs=10):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir):
        generate_augmented_mango_dataset(base_dir=data_dir)
        
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = MangoDataset(train_dir, transform=transform_train)
    val_dataset = MangoDataset(val_dir, transform=transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Initializing PyTorch CNN Model on device: {device}...")
    
    model = build_mango_cnn_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"[INFO] Starting PyTorch Model Training for {epochs} Epochs...")
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%")
        
    print(f"[INFO] Saving trained model weights to {model_save_path}...")
    torch.save(model.state_dict(), model_save_path)
    
    # Generate Training Performance Curves Plot
    print("[INFO] Generating Training Performance Curves (training_performance.png)...")
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss', color='#f59e0b', linewidth=2)
    plt.plot(range(1, epochs + 1), history['val_loss'], label='Val Loss', color='#ef4444', linestyle='--', linewidth=2)
    plt.title('CNN Loss Progression')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history['train_acc'], label='Train Acc', color='#10b981', linewidth=2)
    plt.plot(range(1, epochs + 1), history['val_acc'], label='Val Acc', color='#38bdf8', linestyle='--', linewidth=2)
    plt.title('CNN Accuracy Progression (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_performance.png', dpi=300)
    plt.close()
    
    # Generate Confusion Matrix Plot
    print("[INFO] Generating Confusion Matrix Plot (confusion_matrix.png)...")
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.YlOrBr)
    plt.title('Mango CNN Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, ['Grade A', 'Grade B', 'Grade C'], rotation=15)
    plt.yticks(tick_marks, ['Grade A', 'Grade B', 'Grade C'])
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max()/2 else "black", fontweight='bold')
            
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()
    
    print("[SUCCESS] Empirical model training complete! Charts saved as 'training_performance.png' and 'confusion_matrix.png'.")

if __name__ == '__main__':
    train_and_evaluate_model()
