import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os, random, glob, time, gc, string

# Import from your config.py
from config import CHARS, CAPTCHA_LENGTH, WIDTH, HEIGHT, BATCH_SIZE, DEVICE, DATASET_SIZE

# --- 1. MODEL DEFINITION ---
class CaptchaModel(nn.Module):
    def __init__(self):
        super(CaptchaModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2)
        )
        # For 80x200 input, the size after these 4 MaxPool layers is (256, 5, 12)
        self.fc = nn.Linear(256 * 5 * 12, 1024)
        self.output = nn.Linear(1024, CAPTCHA_LENGTH * len(CHARS))

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.output(x).view(-1, CAPTCHA_LENGTH, len(CHARS))

# --- 2. DATASET DEFINITION ---
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

# --- 3. TRAINING LOOP ---
def train():
    # Native XPU Support for Arrow Lake
    device = torch.device("xpu") 
    model = CaptchaModel().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    criterion = nn.CrossEntropyLoss()
    
    dataset_path = os.path.join(os.getcwd(), "dataset")
    if not os.path.exists(dataset_path):
        print("❌ Dataset folder not found. Please run your generator first!")
        return

    full_ds = CaptchaDataset(dataset_path)
    train_size = int(0.9 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [train_size, len(full_ds)-train_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print(f"🚀 Starting Refinement Training on {torch.xpu.get_device_name(0)}")
    best_acc = 0.0

    for epoch in range(35): # Extended to 35 for the "breakthrough"
        epoch_start = time.time()
        model.train()
        total_loss = 0
        
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            
            # Using torch.amp for mixed precision on XPU
            with torch.amp.autocast(device_type="xpu", dtype=torch.float16):
                outputs = model(imgs)
                loss = criterion(outputs.view(-1, len(CHARS)), lbls.view(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

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
        avg_loss = total_loss / len(train_loader)
        
        print(f"✅ Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Acc: {val_acc:.2f}% | Time: {time.time()-epoch_start:.2f}s")

        # --- SMART STOP ---
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "captcha_model_best.pth")

        if val_acc >= 98.5:
            print("🎯 Target reached. Stopping to save heat.")
            break

        # SFF/Mini PC Maintenance
        torch.xpu.empty_cache()
        gc.collect()
        time.sleep(10) 

    torch.save(model.state_dict(), "captcha_model_final.pth")
    print("✨ Training complete. Model saved as captcha_model_final.pth")

if __name__ == "__main__":
    train()
