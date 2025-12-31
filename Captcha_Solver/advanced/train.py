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

# Capture absolute start - The "Wall Clock" anchor
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
    """Manual math for wall-clock time accuracy."""
    s = int(seconds)
    hours = s // 3600
    minutes = (s % 3600) // 60
    secs = s % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def train():
    dataset = CaptchaDataset(config.DATASET_SIZE, config.CHARS, config.CAPTCHA_LENGTH, config.WIDTH, config.HEIGHT)
    
    # OPTIMIZED: 4 workers prevents the 20GB+ memory commitment seen in your Task Manager.
    dataloader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=2
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
            
            # Aggressive cache clearing to keep XPU memory pressure low
            if i % 100 == 0:
                torch.xpu.empty_cache()
            
            if i % 20 == 0:
                now = datetime.now()
                
                # Total Wall Clock
                total_seconds_elapsed = (now - START_TIME).total_seconds()
                total_str = format_seconds(total_seconds_elapsed)

                # Epoch Speed and ETA
                epoch_elapsed = (now - epoch_start_time).total_seconds()
                it_per_sec = (i + 1) / epoch_elapsed if epoch_elapsed > 0 else 0
                remaining_batches = len(dataloader) - (i + 1)
                eta_secs = remaining_batches / it_per_sec if it_per_sec > 0 else 0
                
                eta_str = f"{int(eta_secs // 60):02d}:{int(eta_secs % 60):02d}"
                
                # Print with trailing spaces to clear any 'ghost' characters
                print(f"Ep {epoch+1:02d} | Loss: {loss.item():.4f} | {it_per_sec:.2f} it/s | ETA: {eta_str} | Total: {total_str}        ", end='\r')

        # End of epoch summary
        epoch_end_total = (datetime.now() - START_TIME).total_seconds()
        print(f"\n✅ Epoch {epoch+1:02d} | Final Loss: {loss.item():.4f} | Total Run Time: {format_seconds(epoch_end_total)}")
        torch.save(model.state_dict(), "advanced_lab_model.pth")

if __name__ == "__main__":
    train()
