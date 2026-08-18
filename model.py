import torch
import torch.nn as nn
import torchvision.models as models

CLASS_NAMES = ['Grade_A_Ripe', 'Grade_B_Unripe', 'Grade_C_Overripe', 'Non_Mango']
CLASS_DISPLAY_NAMES = {
    'Grade_A_Ripe': 'Grade A - Ripe / Fresh 🥭',
    'Grade_B_Unripe': 'Grade B - Unripe / Green 🍏',
    'Grade_C_Overripe': 'Grade C - Overripe / Damaged 🍂',
    'Non_Mango': 'Invalid Object / Not a Mango 🚫'
}

class MangoMobileNetV2(nn.Module):
    """MobileNetV2 Transfer Learning Architecture leveraging 1.4M ImageNet Pre-trained Feature Weights."""
    def __init__(self, num_classes=4, pretrained=True):
        super(MangoMobileNetV2, self).__init__()
        try:
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.base_model = models.mobilenet_v2(weights=weights)
        except Exception:
            self.base_model = models.mobilenet_v2(pretrained=pretrained)
        
        # Freeze base feature extractor layers
        for param in self.base_model.features.parameters():
            param.requires_grad = False
            
        # Replace classifier head for 4 Quality Classes
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

class DeepMangoCNN(nn.Module):
    """Custom 4-Block Deep Convolutional Neural Network Architecture."""
    def __init__(self, num_classes=4):
        super(DeepMangoCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(256 * 4 * 4, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def build_mango_cnn_model(num_classes=4, model_type='mobilenet'):
    """
    Factory function returning model instance.
    - model_type='mobilenet': MobileNetV2 Transfer Learning (Recommended!)
    - model_type='deep_cnn': Custom 4-Block Deep CNN
    """
    if model_type == 'mobilenet':
        print(f"[INFO] Initializing MobileNetV2 Transfer Learning Model (Classes: {num_classes})...")
        return MangoMobileNetV2(num_classes=num_classes)
    else:
        print(f"[INFO] Initializing Custom 4-Block Deep CNN Model (Classes: {num_classes})...")
        return DeepMangoCNN(num_classes=num_classes)

if __name__ == '__main__':
    model = build_mango_cnn_model(model_type='mobilenet')
    dummy_input = torch.randn(1, 3, 224, 224)
    print("MobileNetV2 Output Shape:", model(dummy_input).shape)
