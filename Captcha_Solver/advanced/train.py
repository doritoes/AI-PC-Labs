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
        
    def __getitem__(self, idx):
        # We simplify the generation to the absolute minimum
        target_text = ''.join(np.random.choice(list(self.chars), self.length))
        # Direct generation without extra PIL conversions
        img = self.generator.generate_image(target_text).convert('L')
        img_tensor = transforms.ToTensor()(img)
        
        target = torch.zeros(self.length, len(self.chars))
        for i, char in enumerate(target_text):
            target[i, self.chars.find(char)] = 1
        return img_tensor, target

    def __len__(self): return self.size

def format_seconds(seconds):
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

def train():
    dataset = CaptchaDataset(config.DATASET_SIZE, config.CHARS, config.CAPTCHA_LENGTH, config.WIDTH, config.HEIGHT)
    
    # RADICAL CHANGE: num_workers=0
    # This runs the data generation in the MAIN process. 
    # While it sounds slower, it eliminates the 10GB+ "Committed Memory" overhead of sub-processes.
    dataloader = DataLoader(
        dataset, 
        batch_size=64,     # Increased to 64: Give the GPU a larger chunk to stay busy
        shuffle=True, 
        num_workers=0,      
        pin_memory=True
    )

    device = torch.device(config.DEVICE)
    model = AdvancedCaptchaModel().to(device)
    
    if os.path.exists("advanced_lab_model.pth"):
        model.load_state_dict(torch.load("advanced_lab_model.pth", map_location=device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.LEARNING_RATE, steps_per_epoch=len(dataloader), epochs=config.EPOCHS)

    print(f"🚀 Active | Device: {config.DEVICE} | Mode: Single-Process (Memory Safe)")

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
            
            # Flush cache every 20 batches
            if i % 20 == 0:
                torch.xpu.empty_cache()
                now = datetime.now()
                it_per_sec = (i + 1) / (now - epoch_start).total_seconds()
                eta_secs = (len(dataloader) - (i + 1)) / it_per_sec if it_per_sec > 0 else 0
                
                print(f"Ep {epoch+1:02d} | Loss: {loss.item():.4f} | {it_per_sec:.2f} it/s | Total: {format_seconds((now - START_TIME).total_seconds())}        ", end='\r')

        print(f"\n✅ Epoch {epoch+1:02d} Saved.")
        torch.save(model.state_dict(), "advanced_lab_model.pth")

if __name__ == "__main__":
    train()
