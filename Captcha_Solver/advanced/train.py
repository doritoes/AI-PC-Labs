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

# Capture absolute start
START_TIME = datetime.now()

class CaptchaDataset(Dataset):
    def __init__(self, size, chars, length, width, height, use_fixed_buffer=True):
        self.size = size
        self.chars = chars
        self.length = length
        self.use_fixed_buffer = use_fixed_buffer
        self.generator = ImageCaptcha(width=width, height=height)
        self.buffer = []

        if self.use_fixed_buffer:
            print(f"📦 MODE: Fixed Buffer | Pre-generating {size} images...")
            for i in range(size):
                item = self._generate_sample()
                self.buffer.append(item)
                if (i + 1) % 1000 == 0:
                    print(f"  > Loaded {i + 1}/{size}...")
            print("✅ Buffer Ready.")
        else:
            print("🚀 MODE: Infinite Random | Generating on-the-fly.")

    def _generate_sample(self):
        target_text = ''.join(np.random.choice(list(self.chars), self.length))
        img = self.generator.generate_image(target_text).convert('L')
        img_tensor = transforms.ToTensor()(img)
        
        target = torch.zeros(self.length, len(self.chars))
        for i, char in enumerate(target_text):
            target[i, self.chars.find(char)] = 1
        return img_tensor, target

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.use_fixed_buffer:
            return self.buffer[idx]
        else:
            return self._generate_sample()

def format_seconds(seconds):
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

def train():
    # --- SUCCESS SWITCH ---
    # Set to True to drop loss quickly. Set to False for infinite variety.
    USE_FIXED_BUFFER = True 
    BUFFER_SIZE = 10000 
    # ----------------------

    dataset = CaptchaDataset(
        BUFFER_SIZE, config.CHARS, config.CAPTCHA_LENGTH, 
        config.WIDTH, config.HEIGHT, use_fixed_buffer=USE_FIXED_BUFFER
    )
    
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
        optimizer, max_lr=config.LEARNING_RATE, 
        steps_per_epoch=len(dataloader), epochs=config.EPOCHS
    )

    print(f"🚀 Active | Device: {config.DEVICE} | Started At: {START_TIME.strftime('%H:%M:%S')}")

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
            
            if i % 10 == 0:
                now = datetime.now()
                total_str = format_seconds((now - START_TIME).total_seconds())
                it_per_sec = (i + 1) / (now - epoch_start_time).total_seconds()
                progress = ((i + 1) / len(dataloader)) * 100
                eta_secs = (len(dataloader) - (i + 1)) / it_per_sec if it_per_sec > 0 else 0
                
                print(f"Ep {epoch+1:02d} | {progress:5.1f}% | Loss: {loss.item():.4f} | {it_per_sec:.2f} it/s | ETA: {int(eta_secs // 60):02d}:{int(eta_secs % 60):02d} | Total: {total_str}        ", end='\r')

        torch.xpu.empty_cache()
        print(f"\n✅ Epoch {epoch+1:02d} | Loss: {loss.item():.4f}")
        torch.save(model.state_dict(), "advanced_lab_model.pth")

if __name__ == "__main__":
    train()
