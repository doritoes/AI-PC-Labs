import torch
import torch.nn as nn
import config

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x): 
        # The 'Residual' magic: it adds the input back to the output
        return torch.relu(x + self.conv(x))

class AdvancedCaptchaModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: Initial feature extraction
        self.layer1 = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.res1 = ResidualBlock(64)
        
        # Layer 2: Deeper features
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.res2 = ResidualBlock(128)
        
        # Layer 3: Final convolution
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2))
        
        # Calculation for Flatten: (200/2/2/2=25 width) * (80/2/2/2=10 height) * 256 filters
        self.fc = nn.Sequential(nn.Linear(256 * 10 * 25, 512), nn.ReLU(), nn.Dropout(0.3))
        self.output = nn.Linear(512, config.CAPTCHA_LENGTH * len(config.CHARS))

    def forward(self, x):
        x = self.layer1(x)
        x = self.res1(x)
        x = self.layer2(x)
        x = self.res2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.output(x).view(-1, config.CAPTCHA_LENGTH, len(config.CHARS))
