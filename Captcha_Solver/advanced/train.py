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

# --- CURRICULUM PARAMETERS ---
STAGE_1_EPOCHS = 20  # DIGITS ONLY
STAGE_2_EPOCHS = 40  # FULL ALPHANUM (HYBRID)
DYNAMIC_RATIO = 0.30 # 30% Fresh images per batch
BATCH_SIZE = config.BATCH_SIZE

class CurriculumDataset(Dataset):
    def __init__(self, mode='digits'):
        self.mode = mode
        self.generator = ImageCaptcha(width=config.WIDTH, height=config.HEIGHT)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.buffer_size = 4000 # Reduced slightly for faster startup
        self.static_data = []
        self.refresh_buffer()

    def refresh_buffer(self):
        print(f"\n🔄 [DATA PREP] Generating {self.buffer_size} static {self.mode} anchors...")
        start = time.time()
        chars = config.DIGITS if self.mode == 'digits' else config.CHARS
        for _ in range(self.buffer_size):
            text = ''.join(np.random.choice(list(chars), config.CAPTCHA_LENGTH))
            img = self.generator.generate_image(text).convert('L')
            self.static_data.append((self.transform(img), text))
        print(f"✅ Buffer Ready ({time.time()-start:.1f}s)")

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, idx):
        # 30% Dynamic Injection logic
        is_dynamic = (self.mode == 'full' and np.random.random() < DYNAMIC_RATIO)
        
        if is_dynamic:
            text = ''.join(np.random.choice(list(config.CHARS), config.CAPTCHA_LENGTH))
            img = self.generator.generate_image(text).convert('L')
            img_tensor = self.transform(img)
        else:
            img_tensor, text = self.static_data[idx]
        
        # Multi-label encoding
        target = torch.zeros(config.CAPTCHA_LENGTH, len(config.CHARS))
        for i, char in enumerate(text):
            target[i, config.CHARS.find(char)] = 1
        return img_tensor, target.flatten()

def train():
    device = torch.device(config.DEVICE)
    model = AdvancedCaptchaModel().to(device)
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    print("\n" + "="*50)
    print(f"🎓 LAB COMMENCED | Device: {config.DEVICE}")
    print(f"📡 Plan: {STAGE_1_EPOCHS} Epochs (Digits) -> {STAGE_2_EPOCHS} Epochs (Hybrid)")
    print(f"🧪 Hybridization: {int(DYNAMIC_RATIO*100)}% Fresh Images / {100-int(DYNAMIC_RATIO*100)}% Static")
    print("="*50)

    dataset = CurriculumDataset(mode='digits')
    
    for epoch in range(1, STAGE_1_EPOCHS + STAGE_2_EPOCHS + 1):
        # --- Stage Transition ---
        if epoch == STAGE_1_EPOCHS + 1:
            print("\n" + "!"*50)
            print("🚀 CURRICULUM UPGRADE: Switching to Full Alphanumeric + Hybrid")
            print("!"*50)
            dataset = CurriculumDataset(mode='full')
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0005 # Dropping LR for fine-tuning
        
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        model.train()
        
        epoch_loss = 0
        correct, total = 0, 0
        start_time = time.time()
        
        for imgs, target in dataloader:
            imgs, target = imgs.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Tracking Char-Level Accuracy
            out_reshaped = output.view(-1, 6, 62).argmax(2)
            tar_reshaped = target.view(-1, 6, 62).argmax(2)
            correct += (out_reshaped == tar_reshaped).sum().item()
            total += tar_reshaped.numel()

        avg_loss = epoch_loss / len(dataloader)
        acc = (correct / total) * 100
        
        # --- STATUS UPDATE ---
        phase_label = "DIGITS ONLY" if epoch <= STAGE_1_EPOCHS else f"HYBRID ({int(DYNAMIC_RATIO*100)}% FRESH)"
        print(f"Epoch [{epoch:02d}/{STAGE_1_EPOCHS+STAGE_2_EPOCHS}] | Phase: {phase_label}")
        print(f" ╰─ Accuracy: {acc:>5.2f}% | Loss: {avg_loss:.4f} | Time: {time.time()-start_time:.1f}s")
        
        if epoch % 5 == 0:
            torch.save(model.state_dict(), "advanced_lab_model.pth")
            print(f" 💾 Checkpoint saved.")

if __name__ == "__main__":
    train()
