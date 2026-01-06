import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from captcha.image import ImageCaptcha
import numpy as np
from datetime import datetime
import os
import string
import gc
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

        print(f"📦 DATA PREP: Pre-loading {size} images ({len(chars)} char set)...")
        for i in range(size):
            target_text = ''.join(np.random.choice(list(self.chars), self.length))
            img = self.generator.generate_image(target_text).convert('L')
            img_tensor = transforms.ToTensor()(img)
            
            # Always use 62 for the target tensor width to match model output
            target = torch.zeros(self.length, 62) 
            for i_char, char in enumerate(target_text):
                global_idx = config.CHARS.find(char)
                target[i_char, global_idx] = 1
            
            self.buffer.append((img_tensor, target))
            if (i + 1) % 2000 == 0:
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
    device = torch.device(config.DEVICE)
    model = AdvancedCaptchaModel().to(device)
    
    # --- RESUME LOGIC ---
    # Loads existing weights so you don't lose progress from previous long runs
    if os.path.exists("advanced_lab_model.pth"):
        print("🔄 RESUMING: Loading weights from advanced_lab_model.pth...")
        try:
            model.load_state_dict(torch.load("advanced_lab_model.pth", map_location=device))
        except Exception as e:
            print(f"⚠️ Could not load weights, starting fresh. Error: {e}")

    # --- CURRICULUM SETUP ---
    digits_only = string.digits
    full_set = config.CHARS
    
    # If starting from a resume, check if we should be in Digits or Full set
    # Since you were on Epoch 4, we start with Digits
    current_chars = digits_only
    dataset = CaptchaDataset(config.DATASET_SIZE, current_chars, config.CAPTCHA_LENGTH, config.WIDTH, config.HEIGHT)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # Initialize scheduler - note: if resuming, scheduler state won't match perfectly
    # but OneCycleLR will recalibrate as it ramps up.
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.LEARNING_RATE, 
        steps_per_epoch=len(dataloader), epochs=config.EPOCHS
    )

    print(f"🚀 Training Active | Device: {config.DEVICE} | Started: {START_TIME.strftime('%H:%M:%S')}")

    # START RANGE AT 3 (Epoch 4) to pick up where you left off
    for epoch in range(3, config.EPOCHS):
        
        # Switch to full alphanumeric set at Epoch 6
        if epoch == 5:
            print("\n🎓 CURRICULUM UPGRADE: Switching to Full Alphanumeric Set...")
            current_chars = full_set
            # Clear old data from RAM before loading new set
            del dataset
            del dataloader
            gc.collect()
            
            dataset = CaptchaDataset(config.DATASET_SIZE, current_chars, config.CAPTCHA_LENGTH, config.WIDTH, config.HEIGHT)
            dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

        model.train()
        epoch_start = datetime.now()
        
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Accuracy Calculation (Per Character)
            out_reshaped = outputs.view(-1, 6, 62)
            lbl_reshaped = labels.view(-1, 6, 62).argmax(dim=2)
            preds = out_reshaped.argmax(dim=2)
            correct = (preds == lbl_reshaped).float().mean().item()
            
            # Loss Calculation
            loss = criterion(outputs.view(-1, 62), labels.view(-1, 62).argmax(dim=1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if i % 10 == 0:
                now = datetime.now()
                elapsed = (now - epoch_start).total_seconds()
                it_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                progress = ((i + 1) / len(dataloader)) * 100
                
                print(f"Ep {epoch+1:02d} | {progress:4.1f}% | Acc: {correct:6.1%} | Loss: {loss.item():.4f} | {it_per_sec:.2f} it/s | Total: {format_seconds((now - START_TIME).total_seconds())}    ", end='\r')

        print(f"\n✅ Epoch {epoch+1:02d} | Final Loss: {loss.item():.4f} | Final Acc: {correct:6.1%} | Set: {'Digits' if epoch < 5 else 'Full'}")
        
        # Save progress and clear cache
        torch.save(model.state_dict(), "advanced_lab_model.pth")
        if hasattr(torch, 'xpu'): torch.xpu.empty_cache()
        gc.collect()

if __name__ == "__main__":
    train()
