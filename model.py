import torch
import torch.nn as nn

CLASS_NAMES = ['Grade_A_Ripe', 'Grade_B_Unripe', 'Grade_C_Overripe']
CLASS_DISPLAY_NAMES = {
    'Grade_A_Ripe': 'Grade A - Ripe / Fresh 🥭',
    'Grade_B_Unripe': 'Grade B - Unripe / Green 🍏',
    'Grade_C_Overripe': 'Grade C - Overripe / Damaged 🍂'
}

class MangoCNN(nn.Module):
    """Custom 3-block Convolutional Neural Network architecture."""
    def __init__(self, num_classes=3):
        super(MangoCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def build_mango_cnn_model(num_classes=3):
    """Factory function returning custom MangoCNN model."""
    print("[INFO] Initializing Custom MangoCNN Model...")
    return MangoCNN(num_classes=num_classes)

if __name__ == '__main__':
    model = build_mango_cnn_model()
    dummy_input = torch.randn(1, 3, 224, 224)
    print("MangoCNN Output Shape:", model(dummy_input).shape)
