import torch
import torch.nn as nn

class AdvancedCaptchaModel(nn.Module):
    def __init__(self):
        super(AdvancedCaptchaModel, self).__init__()
        # Layer 1: Enhanced Feature Extraction
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64) # Force normalization
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Layer 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Layer 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fully Connected
        self.dropout = nn.Dropout(0.3) # Prevent memorization
        self.fc = nn.Linear(256 * 25 * 10, 1024)
        self.output = nn.Linear(1024, 6 * 62) # 6 chars * 62 possibilities

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        x = self.dropout(x)
        x = self.output(x)
        return x
