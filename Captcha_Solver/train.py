import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from captcha.image import ImageCaptcha
import numpy as np
from datetime import datetime
import os
import config
from model import AdvancedCaptchaModel

START_TIME = datetime.now()

class CaptchaDataset(Dataset):
    def __init__(self, size, chars, length, width, height):
        self.size = size
        self.chars = chars
        self.length = length
        self.generator = ImageCaptcha(width=width, height=height)
        self.buffer = []

        print(f"📦 DATA PREP: Pre-loading {size} images into RAM...")
        for i in range(size):
            target_text = ''.join(np.random.choice(list(self.chars), self.length))
            img = self.generator.generate_image(target_text).convert('L')
            img_tensor = transforms.ToTensor()(img)
            
            target = torch.zeros(self.length, len(self.chars))
            for i_char, char in enumerate(target_text):
                target[i_char, self.chars.find(char)] = 1
            
            self.buffer.append((img_tensor, target))
            if (i + 1) % 5000 == 0:
                print(f"  > Cached {i + 1}/{size}...")
        print("✅ RAM Buffer Ready.")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.buffer[idx]

def format_seconds(seconds):
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

def train():
    dataset = CaptchaDataset(config.DATASET_SIZE, config.CHARS, config.CAPTCHA_LENGTH, config.WIDTH, config.HEIGHT)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    device = torch.device(config.DEVICE)
    model = AdvancedCaptchaModel().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.LEARNING_RATE, 
        steps_per_epoch=len(dataloader), epochs=config.EPOCHS
    )

    print(f"🚀 Training Active | Device: {config.DEVICE} | Started: {START_TIME.strftime('%H:%M:%S')}")

    for epoch in range(config.EPOCHS):
        model.train()
        epoch_start = datetime.now()
        
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.view(-1, len(config.CHARS)), labels.view(-1, len(config.CHARS)).argmax(dim=1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if i % 10 == 0:
                now = datetime.now()
                elapsed = (now - epoch_start).total_seconds()
                it_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                
                # Progress and ETA
                progress = ((i + 1) / len(dataloader)) * 100
                rem_its = len(dataloader) - (i + 1)
                eta_s = rem_its / it_per_sec if it_per_sec > 0 else 0
                
                print(f"Ep {epoch+1:02d} | {progress:5.1f}% | Loss: {loss.item():.4f} | {it_per_sec:.2f} it/s | ETA: {int(eta_s//60):02d}:{int(eta_s%60):02d} | Total: {format_seconds((now - START_TIME).total_seconds())}    ", end='\r')

        print(f"\n✅ Epoch {epoch+1:02d} | Final Loss: {loss.item():.4f}")
        torch.save(model.state_dict(), "advanced_lab_model.pth")
        if hasattr(torch, 'xpu'): torch.xpu.empty_cache()

if __name__ == "__main__":
    train()
