import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from captcha.image import ImageCaptcha
import numpy as np
import os
import time
import config
from model import AdvancedCaptchaModel

# --- LAB SETTINGS ---
STAGE_1_EPOCHS = 20  # Digits only (Static)
STAGE_2_EPOCHS = 40  # Full Alphanumeric (70% Static / 30% Dynamic)
BATCH_SIZE = 64

class CurriculumDataset(Dataset):
    def __init__(self, mode='digits'):
        self.mode = mode
        self.generator = ImageCaptcha(width=config.WIDTH, height=config.HEIGHT)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        # The 'Anchor' Buffer: 6000 consistent images
        self.buffer_size = 6000
        self.static_data = []
        self.refresh_buffer()

    def refresh_buffer(self):
        print(f"📦 DATA PREP: Refreshing {self.mode} buffer...")
        self.static_data = []
        chars = config.DIGITS if self.mode == 'digits' else config.CHARS
        for _ in range(self.buffer_size):
            text = ''.join(np.random.choice(list(chars), config.CAPTCHA_LENGTH))
            img = self.generator.generate_image(text).convert('L')
            self.static_data.append((self.transform(img), text))

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, idx):
        # 70/30 Hybrid Logic for Stage 2
        if self.mode == 'full' and np.random.random() < 0.30:
            # Generate a fresh "Dynamic" image (The 30%)
            text = ''.join(np.random.choice(list(config.CHARS), config.CAPTCHA_LENGTH))
            img = self.generator.generate_image(text).convert('L')
            img_tensor = self.transform(img)
        else:
            # Use the "Static" buffer (The 70%)
            img_tensor, text = self.static_data[idx]
        
        # Encode label
        target = torch.zeros(config.CAPTCHA_LENGTH, len(config.CHARS))
        for i, char in enumerate(text):
            target[i, config.CHARS.find(char)] = 1
        return img_tensor, target.flatten()

def train():
    device = torch.device(config.DEVICE)
    model = AdvancedCaptchaModel().to(device)
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # 1000 steps per epoch to ensure deep exposure
    steps_per_epoch = 100
    
    print(f"🚀 Training Started: {STAGE_1_EPOCHS} Digits | {STAGE_2_EPOCHS} Hybrid")

    for epoch in range(1, STAGE_1_EPOCHS + STAGE_2_EPOCHS + 1):
        # --- CURRICULUM SWITCHING ---
        if epoch <= STAGE_1_EPOCHS:
            mode = 'digits'
            if epoch == 1: dataset = CurriculumDataset(mode='digits')
        else:
            mode = 'full'
            if epoch == STAGE_1_EPOCHS + 1:
                print("\n🎓 CURRICULUM UPGRADE: Switching to Hybrid Alphanumeric...")
                dataset = CurriculumDataset(mode='full')
                # Lower LR for the fine-tuning phase
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 5e-4

        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        model.train()
        
        epoch_loss = 0
        correct_chars = 0
        total_chars = 0
        
        start_time = time.time()
        
        for i, (imgs, target) in enumerate(dataloader):
            imgs, target = imgs.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Simple Accuracy Tracking
            out = output.view(-1, 6, 62)
            tar = target.view(-1, 6, 62)
            acc = (out.argmax(2) == tar.argmax(2)).float().mean()
            correct_chars += (out.argmax(2) == tar.argmax(2)).sum().item()
            total_chars += tar.size(0) * tar.size(1)

        avg_loss = epoch_loss / len(dataloader)
        char_acc = (correct_chars / total_chars) * 100
        duration = time.time() - start_time
        
        print(f"Ep {epoch:02d} [{mode.upper()}] | Acc: {char_acc:>5.1f}% | Loss: {avg_loss:.4f} | {duration:.1f}s")
        
        # Save checkpoints
        if epoch % 5 == 0 or epoch == (STAGE_1_EPOCHS + STAGE_2_EPOCHS):
            torch.save(model.state_dict(), "advanced_lab_model.pth")
            print(f"💾 Model Saved.")

if __name__ == "__main__":
    train()
