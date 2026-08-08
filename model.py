import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES = ['Grade_A_Ripe', 'Grade_B_Unripe', 'Grade_C_Overripe']

CLASS_DISPLAY_NAMES = {
    'Grade_A_Ripe': 'Grade A - Ripe / Fresh 🥭',
    'Grade_B_Unripe': 'Grade B - Unripe / Green 🍏',
    'Grade_C_Overripe': 'Grade C - Overripe / Damaged 🍂'
}

class MangoCNN(nn.Module):
    """
    Convolutional Neural Network (CNN) in PyTorch for Mango Quality & Ripeness Classification.
    Features: 3 Convolutional Blocks with BatchNorm & MaxPool, followed by Fully Connected Classifier.
    """
    def __init__(self, num_classes=3):
        super(MangoCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Classifier
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: (B, 3, 224, 224)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def build_mango_cnn_model():
    return MangoCNN(num_classes=len(CLASS_NAMES))

if __name__ == "__main__":
    model = build_mango_cnn_model()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print("Model Output Shape:", output.shape)
