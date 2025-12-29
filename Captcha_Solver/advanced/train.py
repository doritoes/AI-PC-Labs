import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image, ImageFilter
import os
import random
import glob
from config import CHARS, CAPTCHA_LENGTH, WIDTH, HEIGHT, BATCH_SIZE, DEVICE, DATASET_SIZE

# --- 1. Data Generation Logic (Integrated) ---
def generate_lab_data():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root_dir, "dataset")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check if we already have data to save time
    existing_files = glob.glob(os.path.join(output_dir, "*.png"))
    if len(existing_files) >= DATASET_SIZE:
        print(f"📊 Dataset already exists ({len(existing_files)} images). Skipping generation.")
        return output_dir

    from captcha.image import ImageCaptcha
    generator = ImageCaptcha(width=WIDTH, height=HEIGHT)
    
    print(f"🎨 Lab Mode: Generating {DATASET_SIZE} base images + Blur/Rotate Augmentations...")
    for i in range(DATASET_SIZE):
        label = "".join(random.choices(CHARS, k=CAPTCHA_LENGTH))
        base_path = os.path.join(output_dir, f"{label}_{i}.png")
        
        # 1. Base Image
        img_pil = generator.generate_image(label)
        img_pil.save(base_path)
        
        # 2. Blur Augmentation (using PIL for speed)
        img_pil.filter(ImageFilter.BLUR).save(base_path.replace(".png", "_blur.png"))
        
        # 3. Rotate Augmentation
        img_pil.rotate(random.uniform(-10, 10), resample=Image.BICUBIC, expand=False).save(base_path.replace(".png", "_rot.png"))
        
        if i % 1000 == 0:
            print(f"Generation Progress: {i}/{DATASET_SIZE}")
    return output_dir

# --- 2. Model Architecture ---
class CaptchaModel(nn.Module):
    def __init__(self):
        super(CaptchaModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(nn.Linear(128 * 25 * 10, 512), nn.ReLU(), nn.Dropout(0.3))
        self.heads = nn.ModuleList([nn.Linear(512, len(CHARS)) for _ in range(CAPTCHA_LENGTH)])

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        x = self.fc(x)
        return [head(x) for head in self.heads]

# --- 3. Dataset Class ---
class CaptchaDataset(Dataset):
    def __init__(self, img_dir):
        self.img_paths = glob.glob(os.path.join(img_dir, "*.png"))
        self.transform = transforms.Compose([
            transforms.Grayscale(), transforms.Resize((HEIGHT, WIDTH)), transforms.ToTensor()
        ])
    def __len__(self): return len(self.img_paths)
    def __getitem__(self, idx):
        path = self.img_paths[idx]
        image = self.transform(Image.open(path))
        label_str = os.path.basename(path)[:CAPTCHA_LENGTH]
        label = torch.tensor([CHARS.find(c) for c in label_str], dtype=torch.long)
        return image, label

# --- 4. Main Lab Execution ---
def train():
    # Phase 1: Generation
    dataset_path = generate_lab_data()

    # Phase 2: Setup
    print(f"🚀 Initializing Arrow Lake XPU: {DEVICE}")
    model = CaptchaModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device=DEVICE)

    full_ds = CaptchaDataset(dataset_path)
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=True)

    # Phase 3: Training Loop
    for epoch in range(20):
        model.train()
        t_loss = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="xpu", dtype=torch.float16):
                outs = model(imgs)
                loss = sum(criterion(outs[j], lbls[:, j]) for j in range(CAPTCHA_LENGTH))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            t_loss += loss.item()

        print(f"Epoch {epoch+1} Complete. Train Loss: {t_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), os.path.join(os.path.dirname(dataset_path), "captcha_model.pth"))
    print("💾 Lab Complete. Model Saved.")

if __name__ == "__main__":
    train()
