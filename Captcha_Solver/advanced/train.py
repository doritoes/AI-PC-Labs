import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from captcha.image import ImageCaptcha
from PIL import Image
import numpy as np
import os
import string
import time

# Import our custom modules
import config
from model import AdvancedCaptchaModel

# --- Dataset Generation ---
class CaptchaDataset(Dataset):
    def __init__(self, size, chars, length, width, height):
        self.size = size
        self.chars = chars
        self.length = length
        self.generator = ImageCaptcha(width=width, height=height)
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate random text
        target_text = ''.join(np.random.choice(list(self.chars), self.length))
        
        # Create image and convert to Grayscale
        img = self.generator.generate_image(target_text).convert('L')
        
        # Transform to Tensor and Normalize
        img_tensor = transforms.ToTensor()(img)
        
        # Encode label (Multi-label One-Hot style)
        target = torch.zeros(self.length, len(self.chars))
        for i, char in enumerate(target_text):
            target[i, self.chars.find(char)] = 1
            
        return img_tensor, target

def train():
    print(f"🎨 Generating {config.DATASET_SIZE} unique captchas in memory...")
    dataset = CaptchaDataset(
        size=config.DATASET_SIZE, 
        chars=config.CHARS, 
        length=config.CAPTCHA_LENGTH,
        width=config.WIDTH,
        height=config.HEIGHT
    )
    
    # Using 4 workers to feed the XPU efficiently
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)

    # Initialize Model on Device
    device = torch.device(config.DEVICE)
    model = AdvancedCaptchaModel().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # Scheduler to help "jump" the 4.12 loss floor
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config.LEARNING_RATE, 
        steps_per_epoch=len(dataloader), 
        epochs=config.EPOCHS
    )

    print(f"🚀 Training Active | Device: {config.DEVICE} | Batch: {config.BATCH_SIZE}")
    print(f"---")

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Reshape for CrossEntropy: (Batch * Length, NumChars)
            loss = criterion(outputs.view(-1, len(config.CHARS)), 
                             labels.view(-1, len(config.CHARS)).argmax(dim=1))
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:
                progress = (i + 1) / len(dataloader) * 100
                print(f"Epoch {epoch+1:02d} | [{i+1}/{len(dataloader)}] {progress:.1f}% | Loss: {loss.item():.4f}", end='\r')

        epoch_loss = running_loss / len(dataloader)
        epoch_time = time.time() - start_time
        
        print(f"✅ Epoch {epoch+1:02d} | Final Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f}s")
        
        # Save checkpoint after every epoch
        torch.save(model.state_dict(), "advanced_lab_model.pth")

if __name__ == "__main__":
    train()
