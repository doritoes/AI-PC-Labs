import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image, ImageFilter
import os, random, glob, time, gc, shutil, datetime

from config import CHARS, CAPTCHA_LENGTH, WIDTH, HEIGHT, BATCH_SIZE, DEVICE, DATASET_SIZE

# --- 1. Model & Dataset ---
class CaptchaModel(nn.Module):
    def __init__(self):
        super(CaptchaModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2)
        )
        # Final CNN feature map size is 12x5 for a 200x80 input
        self.fc = nn.Linear(256 * 12 * 5, 1024)
        self.output = nn.Linear(1024, CAPTCHA_LENGTH * len(CHARS))

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.output(x).view(-1, CAPTCHA_LENGTH, len(CHARS))

class CaptchaDataset(Dataset):
    def __init__(self, img_dir):
        self.img_paths = glob.glob(os.path.join(img_dir, "*.png"))
        self.transform = transforms.Compose([
            transforms.Grayscale(), transforms.Resize((HEIGHT, WIDTH)),
            transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
        ])
    def __len__(self): return len(self.img_paths)
    def __getitem__(self, idx):
        path = self.img_paths[idx]
        image = self.transform(Image.open(path))
        label_str = os.path.basename(path)[:CAPTCHA_LENGTH]
        label = torch.tensor([CHARS.find(c) for c in label_str], dtype=torch.long)
        return image, label

# --- 2. Data Generation ---
def generate_fresh_data(output_dir):
    if os.path.exists(output_dir):
        print(f"🧹 Clearing old data...")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    from captcha.image import ImageCaptcha
    generator = ImageCaptcha(width=WIDTH, height=HEIGHT)
    
    print(f"🎨 Generating {DATASET_SIZE*3} fresh images (Base + Blur + Rotate)...")
    for i in range(DATASET_SIZE):
        label = "".join(random.choices(CHARS, k=CAPTCHA_LENGTH))
        base_path = os.path.join(output_dir, f"{label}_{i}.png")
        img_pil = generator.generate_image(label)
        img_pil.save(base_path)
        img_pil.filter(ImageFilter.BLUR).save(base_path.replace(".png", "_blur.png"))
        img_pil.rotate(random.uniform(-15, 15), fillcolor="white").save(base_path.replace(".png", "_rot.png"))
        if i % 10000 == 0 and i > 0:
            print(f"  > Generated {i*3} images...")

# --- 3. Training & Accuracy Logic ---
def train():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(root_dir, "dataset")
    log_path = os.path.join(root_dir, "results_log.txt")
    
    generate_fresh_data(dataset_path)

    print(f"🚀 Initializing Arrow Lake XPU: {DEVICE}")
    model = CaptchaModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device=DEVICE)

    full_ds = CaptchaDataset(dataset_path)
    train_size = int(0.9 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [train_size, len(full_ds)-train_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    best_acc = 0.0
    try:
        start_time = time.time()
        for epoch in range(15):
            model.train()
            epoch_loss = 0
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type="xpu", dtype=torch.float16):
                    outputs = model(imgs)
                    loss = criterion(outputs.view(-1, len(CHARS)), lbls.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()

            # --- Validation: Strict Accuracy ---
            model.eval()
            correct = 0
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                    with torch.amp.autocast(device_type="xpu", dtype=torch.float16):
                        outputs = model(imgs)
                        preds = torch.argmax(outputs, dim=2)
                        # All 6 chars must match exactly
                        correct += (preds == lbls).all(dim=1).sum().item()
            
            val_acc = (correct / len(val_ds)) * 100
            avg_loss = epoch_loss / len(train_loader)
            print(f"✅ Epoch {epoch+1} | Loss: {avg_loss:.4f} | Perfect Match Acc: {val_acc:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(root_dir, "captcha_model.pth"))
                print(f" ⭐ New Best Model Saved ({val_acc:.2f}%)")

        total_time = (time.time() - start_time) / 60
        with open(log_path, "a") as f:
            f.write(f"\n--- Lab Run: {datetime.datetime.now()} ---\n")
            f.write(f"Total Data: {DATASET_SIZE*3} | Time: {total_time:.2f} min\n")
            f.write(f"Final Loss: {avg_loss:.4f} | Best Accuracy: {best_acc:.2f}%\n")
            f.write("-" * 40 + "\n")

    finally:
        print(f"🧹 Cleanup: Removing {dataset_path}...")
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        print("✨ Lab complete.")

if __name__ == "__main__":
    train()
