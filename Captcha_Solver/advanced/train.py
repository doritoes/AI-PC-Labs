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
    """Helper to format seconds into MM:SS or HH:MM:SS"""
    if seconds < 0: return "00:00"
    if seconds < 3600:
        return time.strftime("%M:%S", time.gmtime(seconds))
    return f"{int(seconds // 3600):02d}:{time.strftime('%M:%S', time.gmtime(seconds % 3600))}"

def train():
    dataset = CaptchaDataset(config.DATASET_SIZE, config.CHARS, config.CAPTCHA_LENGTH, config.WIDTH, config.HEIGHT)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=12, 
        pin_memory=True,
        prefetch_factor=2
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

    print(f"🚀 Active | Device: {config.DEVICE} | Workers: 12")
    print(f"---")

    # Start the Global Timer
    global_start_time = time.time()

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
            
            if i % 20 == 0:
                now = time.time()
                elapsed_epoch = now - epoch_start_time
                total_elapsed = now - global_start_time
                
                it_per_sec = (i + 1) / elapsed_epoch
                
                # Calculate ETA for current epoch
                remaining_batches = len(dataloader) - (i + 1)
                eta_seconds = remaining_batches / it_per_sec
                
                progress = ((i + 1) / len(dataloader)) * 100
                
                print(f"Ep {epoch+1:02d} | {progress:5.1f}% | Loss: {loss.item():.4f} | {it_per_sec:4.2f} it/s | ETA: {format_time(eta_seconds)} | Total: {format_time(total_elapsed)}", end='\r')

        epoch_loss = running_loss / len(dataloader)
        print(f"\n✅ Epoch {epoch+1:02d} | Final Loss: {epoch_loss:.4f} | Epoch Time: {format_time(time.time() - epoch_start_time)} | Running Total: {format_time(time.time() - global_start_time)}")
        torch.save(model.state_dict(), "advanced_lab_model.pth")

if __name__ == "__main__":
    train()
