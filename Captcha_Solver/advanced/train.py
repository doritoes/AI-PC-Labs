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
STAGE_1_EPOCHS = 20  
STAGE_2_EPOCHS = 40  
DYNAMIC_RATIO = 0.40 

class HardenedDataset(Dataset):
    def __init__(self, mode='digits'):
        self.mode = mode
        self.generator = ImageCaptcha(width=config.WIDTH, height=config.HEIGHT)
        self.transform = transforms.Compose([
            transforms.RandomPerspective(distortion_scale=0.15, p=0.4),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.buffer_size = 5000 
        self.static_data = []
        self.refresh_buffer()

    def refresh_buffer(self):
        print(f"\n🔄 [DATA PREP] Generating {self.buffer_size} {self.mode.upper()} anchors...")
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
        is_dynamic = (self.mode == 'full' and np.random.random() < DYNAMIC_RATIO)
        if is_dynamic:
            text = ''.join(np.random.choice(list(config.CHARS), config.CAPTCHA_LENGTH))
            img = self.generator.generate_image(text).convert('L')
            img_tensor = self.transform(img)
        else:
            img_tensor, text = self.static_data[idx]
        
        target = torch.zeros(config.CAPTCHA_LENGTH, len(config.CHARS))
        for i, char in enumerate(text):
            target[i, config.CHARS.find(char)] = 1
        return img_tensor, target.flatten()

def train():
    device = torch.device(config.DEVICE)
    model = AdvancedCaptchaModel().to(device)
    criterion = nn.MultiLabelSoftMarginLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    
    print("\n" + "="*80)
    print(f"🎓 LAB COMMENCED | Device: {config.DEVICE}")
    print(f"📡 Curriculum: {STAGE_1_EPOCHS} Ep (Digits) -> {STAGE_2_EPOCHS} Ep (Alphanum)")
    print(f"🧪 Hardening: Perspective Warp + 40% Hybrid Injection")
    print("="*80)

    dataset = HardenedDataset(mode='digits')
    
    for epoch in range(1, STAGE_1_EPOCHS + STAGE_2_EPOCHS + 1):
        if epoch == STAGE_1_EPOCHS + 1:
            print("\n" + "!"*80)
            print("🚀 UPGRADE: Switching to Full Alphanumeric Curriculum")
            print("!"*80)
            dataset = HardenedDataset(mode='full')
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.LEARNING_RATE * 0.5 
        
        dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        model.train()
        
        epoch_loss, correct, total = 0, 0, 0
        start_time = time.time()
        num_batches = len(dataloader)
        
        for batch_idx, (imgs, target) in enumerate(dataloader):
            imgs, target = imgs.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            out_reshaped = output.view(-1, 6, 62).argmax(2)
            tar_reshaped = target.view(-1, 6, 62).argmax(2)
            correct += (out_reshaped == tar_reshaped).sum().item()
            total += tar_reshaped.numel()

            if batch_idx % 5 == 0 or batch_idx == num_batches - 1:
                elapsed = time.time() - start_time
                it_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                acc = (correct / total) * 100
                # RESTORED HUD STRING:
                print(f"\r  ⚡ [Ep {epoch:02d}] Acc: {acc:5.1f}% | Loss: {loss.item():.4f} | {it_per_sec:.2f} it/s   ", end="", flush=True)

        avg_loss = epoch_loss / num_batches
        print(f"\n✅ Epoch [{epoch:02d}] COMPLETE | Final Acc: {(correct/total)*100:.2f}% | Time: {time.time()-start_time:.1f}s")
        
        if epoch % 5 == 0:
            torch.save(model.state_dict(), "advanced_lab_model.pth")

if __name__ == "__main__":
    train()
