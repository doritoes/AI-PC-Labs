import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from captcha.image import ImageCaptcha
import numpy as np
import os

# --- 1. CONFIGURATION ---
CHARS = "0123456789" 
CAPTCHA_LENGTH = 4
WIDTH, HEIGHT = 160, 60
DATASET_SIZE = 10000 
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8 # 80% train, 20% held back
EPOCHS = 10

# --- 2. DATASET GENERATOR ---
class CaptchaDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.generator = ImageCaptcha(width=WIDTH, height=HEIGHT)
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        label_str = "".join([np.random.choice(list(CHARS)) for _ in range(CAPTCHA_LENGTH)])
        img = self.generator.generate_image(label_str)
        # Convert to Grayscale, Normalize to 0-1, and convert to Tensor
        img = np.array(img.convert('L')) / 255.0
        img = torch.FloatTensor(img).unsqueeze(0) 
        
        # Multi-hot encoding for the characters
        label = torch.zeros(CAPTCHA_LENGTH, len(CHARS))
        for i, char in enumerate(label_str):
            label[i][CHARS.find(char)] = 1
        
        return img, label.flatten()

# --- 3. PREPARE DATA ---
print(f"Generating dataset of {DATASET_SIZE} images in memory...")
full_dataset = CaptchaDataset(DATASET_SIZE)
train_size = int(TRAIN_SPLIT * DATASET_SIZE)
test_size = DATASET_SIZE - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Ready: {train_size} training images, {test_size} held-back test images.")

# --- 4. THE MODEL ARCHITECTURE ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2))
        # Image is 160x60 -> after two 2x2 pools, it is 40x15
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

print("\nStarting training on CPU (20 threads)...")
model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f}")

# --- 6. VALIDATION (The Final Exam) ---
print("\nTesting model against held-back images...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        # Reshape output to (Batch, Length, Chars) to check each digit
        output_reshaped = outputs.view(-1, CAPTCHA_LENGTH, len(CHARS))
        labels_reshaped = labels.view(-1, CAPTCHA_LENGTH, len(CHARS))
        
        # Get the index of the max value (the predicted digit)
        pred_digits = output_reshaped.argmax(dim=2)
        true_digits = labels_reshaped.argmax(dim=2)
        
        # A captcha is "solved" only if ALL 4 digits are correct
        correct_indices = (pred_digits == true_digits).all(dim=1)
        correct += correct_indices.sum().item()
        total += labels.size(0)

print(f"Final Accuracy: {(correct/total)*100:.2f}%")

# --- 7. SAVE MODEL ---
save_path = "captcha_model.pth"
try:
    torch.save(model.state_dict(), save_path)
    print(f"\nSUCCESS: Model saved as {save_path}")
    print("You are now ready for Day 2: NPU Optimization.")
except Exception as e:
    print(f"\nERROR: Could not save model: {e}")
