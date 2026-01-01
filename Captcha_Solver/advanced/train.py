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

def format_seconds(seconds):
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

def train():
    dataset = CaptchaDataset(config.DATASET_SIZE, config.CHARS, config.CAPTCHA_LENGTH, config.WIDTH, config.HEIGHT)
    
    # Increase batch size to 128 to maximize GPU usage per CPU cycle
    dataloader = DataLoader(
        dataset, 
        batch_size=128, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )

    device = torch.device(config.DEVICE)
    model = AdvancedCaptchaModel().to(device)
    
    if os.path.exists("advanced_lab_model.pth"):
        try:
            model.load_state_dict(torch.load("advanced_lab_model.pth", map_location=device))
            print("🔄 Loaded existing checkpoint.")
        except:
            pass

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config.LEARNING_RATE, 
        steps_per_epoch=len(dataloader), 
        epochs=config.EPOCHS
    )

    print(f"🚀 Active | Device: {config.DEVICE} | Started: {START_TIME.strftime('%H:%M:%S')}")

    for epoch in range(config.EPOCHS):
        model.train()
        epoch_start_time = datetime.now()
        
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.view(-1, len(config.CHARS)), labels.view(-1, len(config.CHARS)).argmax(dim=1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Wipe cache every 10 iterations (Aggressive for shared iGPU RAM)
            if i % 10 == 0:
                torch.xpu.empty_cache()
                now = datetime.now()
                total_str = format_seconds((now - START_TIME).total_seconds())
                epoch_elapsed = (now - epoch_start_time).total_seconds()
                it_per_sec = (i + 1) / epoch_elapsed if epoch_elapsed > 0 else 0
                
                progress = ((i + 1) / len(dataloader)) * 100
                remaining_batches = len(dataloader) - (i + 1)
                eta_secs = remaining_batches / it_per_sec if it_per_sec > 0 else 0
                eta_str = f"{int(eta_secs // 60):02d}:{int(eta_secs % 60):02d}"
                
                print(f"Ep {epoch+1:02d} | {progress:5.1f}% | Loss: {loss.item():.4f} | {it_per_sec:.2f} it/s | ETA: {eta_str} | Total: {total_str}        ", end='\r')

        print(f"\n✅ Epoch {epoch+1:02d} | Loss: {loss.item():.4f} | Total: {format_seconds((datetime.now() - START_TIME).total_seconds())}")
        torch.save(model.state_dict(), "advanced_lab_model.pth")

if __name__ == "__main__":
    train()
