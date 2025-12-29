import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image, ImageFilter
import os, random, glob, time, gc, string

# Import from your config.py
from config import CHARS, CAPTCHA_LENGTH, WIDTH, HEIGHT, BATCH_SIZE, DATASET_SIZE

# --- 1. DATA GENERATION LOGIC WITH RESTORED TIMERS ---
def prepare_dataset(output_dir):
    gen_start = time.time()
    # Check if dataset already exists and is populated
    if os.path.exists(output_dir) and len(glob.glob(os.path.join(output_dir, "*.png"))) > 100:
        print(f"📊 Dataset already exists. Proceeding to training.")
        return

    from captcha.image import ImageCaptcha
    generator = ImageCaptcha(width=WIDTH, height=HEIGHT)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🎨 Generating {DATASET_SIZE} base images + augmentations...")
    
    for i in range(DATASET_SIZE):
        label = "".join(random.choices(CHARS, k=CAPTCHA_LENGTH))
        base_path = os.path.join(output_dir, f"{label}_{i}.png")
        img_pil = generator.generate_image(label)
        img_pil.save(base_path)
        
        # Fast Augmentations for better learning
        img_pil.filter(ImageFilter.BLUR).save(base_path.replace(".png", "_b.png"))
        img_pil.rotate(random.uniform(-10, 10), fillcolor="white").save(base_path.replace(".png", "_r.png"))
        
        if i % 5000 == 0 and i > 0:
            print(f"  > {i*3} images ready...")

    gen_duration = time.time() - gen_start
    print(f"✅ Generation complete in {gen_duration:.2f}s")

# --- 2. MODEL DEFINITION ---
class CaptchaModel(nn.Module):
    def __init__(self):
        super(CaptchaModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2)
        )
        # Based on 80x200 input, output of conv is (256, 5, 12)
        self.fc = nn.Linear(256 * 5 * 12, 1024)
        self.output = nn.Linear(1024, CAPTCHA_LENGTH * len(CHARS))

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.output(x).view(-1, CAPTCHA_LENGTH, len(CHARS))

# --- 3. DATASET LOADER ---
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

# --- 4. MAIN TRAINING FUNCTION ---
def train():
    dataset_path = os.path.join(os.getcwd(), "dataset")
    prepare_dataset(dataset_path)

    device = torch.device("xpu") 
    model = CaptchaModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    criterion = nn.CrossEntropyLoss()
    
    full_ds = CaptchaDataset(dataset_path)
    train_size = int(0.9 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [train_size, len(full_ds)-train_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print(f"🚀 Training on {torch.xpu.get_device_name(0)} (Max 35 Epochs)")
    best_acc = 0.0

    for epoch in range(35):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="xpu", dtype=torch.float16):
                outputs = model(imgs)
                loss = criterion(outputs.view(-1, len(CHARS)), lbls.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation phase
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
        epoch_duration = time.time() - epoch_start
        print(f"✅ Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Acc: {val_acc:.2f}% | Time: {epoch_duration:.2f}s")

        # Smart Stop to save the best weights
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "captcha_model_best.pth")

        # Auto-stop trigger for the "Refinement Phase"
        if val_acc >= 98.5:
            print("🎯 Target Accuracy reached. Stopping to save heat.")
            break

        # SFF/Mini PC Maintenance
        torch.xpu.empty_cache()
        gc.collect()
        time.sleep(10) # 10s cooldown between epochs for the EliteDesk chassis

    torch.save(model.state_dict(), "captcha_model_final.pth")
    print("✨ Training complete. Model saved as captcha_model_final.pth")

if __name__ == "__main__":
    train()
