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

# The "Wall Clock" start time - Simple and direct.
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

def train():
    dataset = CaptchaDataset(config.DATASET_SIZE, config.CHARS, config.CAPTCHA_LENGTH, config.WIDTH, config.HEIGHT)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

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
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.LEARNING_RATE, steps_per_epoch=len(dataloader), epochs=config.EPOCHS)

    print(f"🚀 Active | Device: {config.DEVICE} | Start: {START_TIME.strftime('%H:%M:%S')}")

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
            
            if i % 20 == 0:
                now = datetime.now()
                # Total time since the script started
                total_elapsed = str(now - START_TIME).split('.')[0] 
                # Time for current epoch
                epoch_elapsed = (now - epoch_start).total_seconds()
                
                it_per_sec = (i + 1) / epoch_elapsed if epoch_elapsed > 0 else 0
                remaining_batches = len(dataloader) - (i + 1)
                eta_secs = remaining_batches / it_per_sec if it_per_sec > 0 else 0
                
                # Simple integer math for ETA MM:SS
                eta = f"{int(eta_secs // 60):02d}:{int(eta_secs % 60):02d}"
                
                print(f"Ep {epoch+1:02d} | Loss: {loss.item():.4f} | {it_per_sec:.2f} it/s | ETA: {eta} | Total: {total_elapsed}", end='\r')

        print(f"\n✅ Epoch {epoch+1:02d} Complete | Total Run Time: {str(datetime.now() - START_TIME).split('.')[0]}")
        torch.save(model.state_dict(), "advanced_lab_model.pth")

if __name__ == "__main__":
    train()
