import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from model import build_mango_cnn_model, CLASS_NAMES
from train import MangoDataset
import time

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = MangoDataset('dataset/val', transform=val_transforms)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_mango_cnn_model(num_classes=4, model_type='efficientnet').to(device)
model.load_state_dict(torch.load('mango_model.pth', map_location=device))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"RAW CNN VALIDATION ACCURACY: {correct/total*100:.2f}%")
