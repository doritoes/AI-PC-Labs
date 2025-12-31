import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os, random, glob, time, gc, sys

# Config Overrides for the Advanced Scenario
from config import CHARS, CAPTCHA_LENGTH, WIDTH, HEIGHT, DATASET_SIZE
BATCH_SIZE = 16 # Increased slightly now that we have deeper architecture
MAX_LR = 0.01

# --- 1. ADVANCED MODEL (ResNet-Lite) ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x): return torch.relu(x + self.conv(x))

class AdvancedCaptchaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.res1 = ResidualBlock(64)
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.res2 = ResidualBlock(128)
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2))
        
        # Flatten and Fully Connected
        self.fc = nn.Sequential(nn.Linear(256 * 10 * 25, 512), nn.ReLU(), nn.Dropout(0.3))
        self.output = nn.Linear(512, CAPTCHA_LENGTH * len(CHARS))

    def forward(self, x):
        x = self.layer1(x)
        x = self.res1(x)
        x = self.layer2(x)
        x = self.res2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.output(x).view(-1, CAPTCHA_LENGTH, len(CHARS))

# --- 2. DATASET LOADER ---
class CaptchaDataset(Dataset):
    def __init__(self, img_dir):
        self.img_paths = glob.glob(os.path.join(img_dir, "*.png"))
        self.transform = transforms.Compose([
            transforms.Grayscale(), 
            transforms.Resize((HEIGHT, WIDTH)),
            transforms.ToTensor(), 
            transforms.Normalize((0.5,), (0.5,))
        ])
    def __len__(self): return len(self.img_paths)
    def __getitem__(self, idx):
        path = self.img_paths[idx]
        image = self.transform(Image.open(path))
        label_str = os.path.basename(path)[:CAPTCHA_LENGTH]
        label = torch.tensor([CHARS.find(c) for c in label_str], dtype=torch.long)
        return image, label

def train():
    dataset_path = os.path.join(os.getcwd(), "dataset")
    device = torch.device("xpu") 
    
    # 20 CPU threads for pre-fetching (num_workers=4 is a safe balance)
    full_ds = CaptchaDataset(dataset_path)
    train_size = int(0.9 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [train_size, len(full_ds)-train_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4)

    model = AdvancedCaptchaModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR/10, weight_decay=0.01)
    
    # OneCycleLR: The "Super-Convergence" Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, 
                                            steps_per_epoch=len(train_loader), epochs=30)
    criterion = nn.CrossEntropyLoss()
    
    print(f"🔬 Advanced Lab Started | Hardware: iGPU + {os.cpu_count()} CPU Threads")

    for epoch in range(30):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        
        for i, (imgs, lbls) in enumerate(train_loader):
            batch_start = time.time()
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type="xpu", dtype=torch.float16):
                outputs = model(imgs)
                loss = criterion(outputs.view(-1, len(CHARS)), lbls.view(-1))
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if i % 25 == 0:
                percent = (i / len(train_loader)) * 100
                sys.stdout.write(f"\rEpoch {epoch+1:02d} | [{'#' * int(percent//5)}{'-' * (20 - int(percent//5))}] {percent:.1f}% | Loss: {loss.item():.4f}")
                sys.stdout.flush()

            # Watchdog Purge
            if (time.time() - batch_start) > 3.0:
                torch.xpu.empty_cache()

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=2)
                correct += (preds == lbls).all(dim=1).sum().item()
        
        val_acc = (correct / len(val_ds)) * 100
        print(f"\n✅ Epoch {epoch+1:02d} | Avg Loss: {total_loss/len(train_loader):.4f} | Acc: {val_acc:.2f}% | Time: {time.time()-epoch_start:.2f}s")
        
        torch.save(model.state_dict(), "advanced_lab_model.pth")
        torch.xpu.empty_cache()
        gc.collect()

if __name__ == "__main__":
    train()
