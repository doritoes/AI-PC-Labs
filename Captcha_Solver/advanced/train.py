import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image, ImageFilter
import os, random, glob, time, gc

from config import CHARS, CAPTCHA_LENGTH, WIDTH, HEIGHT, BATCH_SIZE, DEVICE, DATASET_SIZE

class CaptchaModel(nn.Module):
    def __init__(self):
        super(CaptchaModel, self).__init__()
        # Deeper CNN for better feature extraction
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2)
        )
        # 200/16 = 12.5, 80/16 = 5 -> 256 * 12 * 5 = 15360
        self.fc = nn.Linear(256 * 12 * 5, 1024)
        self.output = nn.Linear(1024, CAPTCHA_LENGTH * len(CHARS))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        x = self.output(x)
        return x.view(-1, CAPTCHA_LENGTH, len(CHARS))

class CaptchaDataset(Dataset):
    def __init__(self, img_dir):
        self.img_paths = glob.glob(os.path.join(img_dir, "*.png"))
        self.transform = transforms.Compose([
            transforms.Grayscale(), transforms.Resize((HEIGHT, WIDTH)),
            transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) # Normalize helps convergence
        ])
    def __len__(self): return len(self.img_paths)
    def __getitem__(self, idx):
        path = self.img_paths[idx]
        image = self.transform(Image.open(path))
        label_str = os.path.basename(path)[:CAPTCHA_LENGTH]
        label = torch.tensor([CHARS.find(c) for c in label_str], dtype=torch.long)
        return image, label

def train():
    # --- Check for dataset folder ---
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(root_dir, "dataset")
    
    if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) < 1000:
        print("❌ Dataset not found or incomplete. Please check generation logic.")
        return

    model = CaptchaModel().to(DEVICE)
    # Lower LR to 0.0005 for stability on Intel XPU
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device=DEVICE)

    full_ds = CaptchaDataset(dataset_path)
    train_ds, _ = random_split(full_ds, [int(0.9*len(full_ds)), len(full_ds)-int(0.9*len(full_ds))])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print(f"🔥 Training Restarted on {DEVICE}. Memory Cleared.")

    for epoch in range(20):
        model.train()
        # CLEAR CACHE EVERY EPOCH
        if DEVICE == "xpu":
            torch.xpu.empty_cache()
        gc.collect()
        
        epoch_loss = 0
        for i, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type="xpu", dtype=torch.float16):
                outputs = model(imgs) # (B, 6, 62)
                loss = criterion(outputs.view(-1, len(CHARS)), lbls.view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

            if i % 100 == 0:
                print(f"E{epoch+1} B{i} | Loss: {loss.item():.4f}")

        print(f"✅ Epoch {epoch+1} Avg Loss: {epoch_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), "../captcha_model.pth")

if __name__ == "__main__":
    train()
