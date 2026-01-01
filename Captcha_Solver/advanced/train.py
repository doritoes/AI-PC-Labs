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
        target_text = ''.join(np.random.choice(list(self.chars), self.length))
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
    
    # 2 WORKERS ONLY: This is the key. 4-8 workers are causing the 16GB swap. 
    # 2 workers will keep "Committed Memory" below 15GB, stopping the SSD lag.
    dataloader = DataLoader(
        dataset, 
        batch_size=32,      # Increased from 16 to 32 to give the GPU more to do per "trip"
        shuffle=True, 
        num_workers=2,      
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=2
    )

    device = torch.device(config.DEVICE)
    model = AdvancedCaptchaModel().to(device)
    
    if os.path.exists("advanced_lab_model.pth"):
        model.load_state_dict(torch.load("advanced_lab_model.pth", map_location=device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.LEARNING_RATE, steps_per_epoch=len(dataloader), epochs=config.EPOCHS)

    print(f"🚀 Active | Device: {config.DEVICE} | Workers: 2 | Started: {START_TIME.strftime('%H:%M:%S')}")

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
            
            if i % 50 == 0: torch.xpu.empty_cache() # More frequent cache clearing
            
            if i % 10 == 0:
                now = datetime.now()
                it_per_sec = (i + 1) / (now - epoch_start).total_seconds()
                eta_secs = (len(dataloader) - (i + 1)) / it_per_sec if it_per_sec > 0 else 0
                
                print(f"Ep {epoch+1:02d} | Loss: {loss.item():.4f} | {it_per_sec:.2f} it/s | ETA: {int(eta_secs // 60):02d}:{int(eta_secs % 60):02d} | Total: {format_seconds((now - START_TIME).total_seconds())}        ", end='\r')

        print(f"\n✅ Epoch {epoch+1:02d} | Loss: {loss.item():.4f}")
        torch.save(model.state_dict(), "advanced_lab_model.pth")

if __name__ == "__main__":
    train()
