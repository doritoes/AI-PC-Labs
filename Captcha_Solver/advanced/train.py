import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from captcha.image import ImageCaptcha
import numpy as np
import time
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
        # Generate on-the-fly (Uses CPU threads)
        target_text = ''.join(np.random.choice(list(self.chars), self.length))
        img = self.generator.generate_image(target_text).convert('L')
        img_tensor = transforms.ToTensor()(img)
        
        target = torch.zeros(self.length, len(self.chars))
        for i, char in enumerate(target_text):
            target[i, self.chars.find(char)] = 1
            
        return img_tensor, target

def train():
    dataset = CaptchaDataset(config.DATASET_SIZE, config.CHARS, config.CAPTCHA_LENGTH, config.WIDTH, config.HEIGHT)
    
    # --- THREAD OPTIMIZATION ---
    # num_workers=12 leverages your 20-thread CPU
    # pin_memory=True speeds up the transfer from RAM to iGPU
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
    
    # Load existing weights if they exist to prevent losing progress
    if os.path.exists("advanced_lab_model.pth"):
        try:
            model.load_state_dict(torch.load("advanced_lab_model.pth", map_location=device))
            print("🔄 Loaded existing checkpoint. Continuing training...")
        except:
            print("⚠️ No valid checkpoint found, starting fresh.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config.LEARNING_RATE, 
        steps_per_epoch=len(dataloader), 
        epochs=config.EPOCHS
    )

    print(f"🚀 Active | Device: {config.DEVICE} | Workers: 12 | Batch: {config.BATCH_SIZE}")

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs.view(-1, len(config.CHARS)), 
                             labels.view(-1, len(config.CHARS)).argmax(dim=1))
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            
            if i % 50 == 0:
                elapsed = time.time() - start_time
                # Added "it/s" (iterations per second) so we can see the speed boost
                it_per_sec = (i + 1) / elapsed
                print(f"Epoch {epoch+1:02d} | [{i}/{len(dataloader)}] Loss: {loss.item():.4f} | {it_per_sec:.2f} it/s", end='\r')

        epoch_loss = running_loss / len(dataloader)
        print(f"\n✅ Epoch {epoch+1:02d} | Final Loss: {epoch_loss:.4f} | Total Time: {time.time() - start_time:.2f}s")
        
        # Save checkpoint
        torch.save(model.state_dict(), "advanced_lab_model.pth")

if __name__ == "__main__":
    import os
    train()
