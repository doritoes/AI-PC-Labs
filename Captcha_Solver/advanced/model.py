import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return torch.relu(out)

class AdvancedCaptchaModel(nn.Module):
    def __init__(self):
        super(AdvancedCaptchaModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual Stack
        self.layer1 = ResidualBlock(64, 128, stride=2)  # Down to 100x40
        self.layer2 = ResidualBlock(128, 256, stride=2) # Down to 50x20
        self.layer3 = ResidualBlock(256, 512, stride=2) # Down to 25x10
        
        self.dropout = nn.Dropout(0.3)
        # 512 filters * 25 width * 10 height
        self.fc = nn.Linear(512 * 25 * 10, 1024)
        self.output = nn.Linear(1024, 6 * 62)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        x = self.dropout(x)
        x = self.output(x)
        return x
