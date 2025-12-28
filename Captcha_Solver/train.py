import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from captcha.image import ImageCaptcha
import numpy as np
import cv2
import os

# --- 1. CONFIGURATION ---
CHARS = "0123456789" # Simple numeric CAPTCHA for speed
CAPTCHA_LENGTH = 4
WIDTH, HEIGHT = 160, 60
DATASET_SIZE = 5000 
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8 # 80% to train, 20% held back for testing

# --- 2. DATASET GENERATOR ---
class CaptchaDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.generator = ImageCaptcha(width=WIDTH, height=HEIGHT)
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate random 4-digit string
        label_str = "".join([np.random.choice(list(CHARS)) for _ in range(CAPTCHA_LENGTH)])
        
        # Create image and convert to grayscale/tensor
        img = self.generator.generate_image(label_str)
        img = np.array(img.convert('L')) / 255.0 # Grayscale and Normalize
        img = torch.FloatTensor(img).unsqueeze(0) # Add channel dimension
        
        # Encode label (Multi-hot encoding)
        label = torch.zeros(CAPTCHA_LENGTH, len(CHARS))
        for i, char in enumerate(label_str):
            label[i][CHARS.find(char)] = 1
        
        return img, label.flatten()

# --- 3. PREPARE DATA ---
full_dataset = CaptchaDataset(DATASET_SIZE)
train_size = int(TRAIN_SPLIT * DATASET_SIZE)
test_size = DATASET_SIZE - train_size

# Split the data so the model "holds back" 1,000 images
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Dataset ready: {train_size} training images, {test_size} held-back test images.")

# --- 4. THE MODEL ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.fc = nn.Linear(64 * 40 * 15, 1024)
        self.output = nn.Linear(1024, CAPTCHA_LENGTH * len(CHARS))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.output(x)

# --- 5. TRAINING LOOP ---
model = SimpleCNN()
criterion = nn.MultiLabelSoftMarginLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting training on CPU...")
model.train()
for epoch in range(5): # 5 Epochs is enough for a lab
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete.")

# --- 6. SAVE MODEL ---
torch.save(model.state_dict(), "captcha_model.pth")
print("Model saved as captcha_model.pth")
