import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from captcha.image import ImageCaptcha
import numpy as np
import time
import os
import config
from model import AdvancedCaptchaModel

# Capture absolute start
WALL_CLOCK_START = time.time()

class CaptchaDataset(Dataset):
    def __init__(self, size, chars, length, width, height):
        self.size = size
        self.chars = chars
        self.length = length
        self.generator = ImageCaptcha(width=width, height=height)
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        target_text = ''.join(np.random.choice(list(self.chars), self.length))
        img = self.generator.generate_image(target_text).convert('L')
        img_tensor = transforms.ToTensor()(img)
        target = torch.zeros(self.length, len(self.chars))
        for i, char in enumerate(target_text):
            target[i, self.chars.find(char)] = 1
        return img_tensor, target

def format_time(seconds):
    """Direct math calculation to prevent '1 sec = 1 min' errors"""
    if seconds is None or seconds < 0: 
        return "00:00"
    
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def train():
    dataset = CaptchaDataset(config.DATASET_SIZE, config.CHARS, config.CAPTCHA_LENGTH, config.WIDTH, config.HEIGHT)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    device = torch.device(config.DEVICE)
    model = AdvancedCaptchaModel().to(device)
    
    if os.path.exists("advanced_lab_model.pth"):
        try:
            model.load_state_dict(torch.load("advanced_lab_model.pth", map_location=device))
            print("🔄 Loaded existing checkpoint.")
        except:
            print("⚠️ Starting fresh.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.LEARNING_RATE, steps_per_epoch=len(dataloader), epochs=config.EPOCHS)

    print(f"🚀 Active | Device: {config.DEVICE} | Workers: 8")
    print(f"---")

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.view(-1, len(config.CHARS)), labels.view(-1, len(config.CHARS)).argmax(dim=1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            
            if i % 500 == 0:
                torch.xpu.empty_cache()
            
            if i % 20 == 0:
                now = time.time()
                elapsed_epoch = now - epoch_start_time
                total_wall_time = now - WALL_CLOCK_START
                
                # it_per_sec calculation
                it_per_sec = (i + 1) / elapsed_epoch if elapsed_epoch > 0 else 0
                
                # ETA calculation
                remaining_batches = len(dataloader) - (i + 1)
                eta_seconds = remaining_batches / it_per_sec if it_per_sec > 0 else 0
                
                progress = ((i + 1) / len(dataloader)) * 100
                
                print(f"Ep {epoch+1:02d} | {progress:5.1f}% | Loss: {loss.item():.4f} | {it_per_sec:4.2f} it/s | ETA: {format_time(eta_seconds)} | Total: {format_time(total_wall_time)}", end='\r')

        epoch_loss = running_loss / len(dataloader)
        print(f"\n✅ Epoch {epoch+1:02d} | Final Loss: {epoch_loss:.4f} | Epoch Time: {format_time(time.time() - epoch_start_time)} | Wall Total: {format_time(time.time() - WALL_CLOCK_START)}")
        torch.save(model.state_dict(), "advanced_lab_model.pth")

if __name__ == "__main__":
    train()
