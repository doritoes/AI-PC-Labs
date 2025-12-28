import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from captcha.image import ImageCaptcha
import numpy as np

# --- 1. CONFIGURATION ---
CHARS = "0123456789"
CAPTCHA_LENGTH = 4
WIDTH, HEIGHT = 160, 60
DATASET_SIZE = 10000
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8 
EPOCHS = 10  # Now using a global constant

# --- 2. HARDWARE DETECTION ---
# Dynamically get the number of logical threads
SYSTEM_THREADS = os.cpu_count() or 1

# --- 3. DATASET GENERATOR ---
class CaptchaDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.generator = ImageCaptcha(width=WIDTH, height=HEIGHT)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        label_str = "".join([np.random.choice(list(CHARS)) for _ in range(CAPTCHA_LENGTH)])
        img = self.generator.generate_image(label_str)
        img = np.array(img.convert('L')) / 255.0
        img = torch.FloatTensor(img).unsqueeze(0)        
        label = torch.zeros(CAPTCHA_LENGTH, len(CHARS))
        for i, char in enumerate(label_str):
            label[i][CHARS.find(char)] = 1
        return img, label.flatten()

# --- 4. PREPARE DATA ---
print(f"System detected: {SYSTEM_THREADS} logical threads.")
print(f"Generating dataset of {DATASET_SIZE} images in memory...")

full_dataset = CaptchaDataset(DATASET_SIZE)
train_size = int(TRAIN_SPLIT * DATASET_SIZE)
test_size = DATASET_SIZE - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# We use num_workers to tell PyTorch to use the system threads for data loading
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Ready: {train_size} training images, {test_size} held-back test images.")

# --- 5. THE MODEL ARCHITECTURE ---
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

# --- 6. TRAINING LOOP ---
model = SimpleCNN()
criterion = nn.MultiLabelSoftMarginLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"\nStarting training on CPU using {SYSTEM_THREADS} threads...")
start_time = time.perf_counter()

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
end_time = time.perf_counter()
total_train_time = end_time - start_time

# --- 7. VALIDATION ---
print("\nTesting model against held-back images...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        output_reshaped = outputs.view(-1, CAPTCHA_LENGTH, len(CHARS))
        labels_reshaped = labels.view(-1, CAPTCHA_LENGTH, len(CHARS))
        pred_digits = output_reshaped.argmax(dim=2)
        true_digits = labels_reshaped.argmax(dim=2)
        correct_indices = (pred_digits == true_digits).all(dim=1)
        correct += correct_indices.sum().item()
        total += labels.size(0)

# --- 8. FINAL SUMMARY & SAVE ---
print("-" * 40)
print("TRAINING SESSION SUMMARY")
print(f"Hardware Threads Used: {SYSTEM_THREADS}")
print(f"Total Training Time:   {total_train_time:.2f} seconds")
print(f"Final Accuracy Score:  {(correct/total)*100:.2f}%")
print("-" * 40)

save_path = "captcha_model.pth"
try:
    torch.save(model.state_dict(), save_path)
    print(f"SUCCESS: Model saved as {save_path}")
except Exception as e:
    print(f"ERROR: Could not save model: {e}")
