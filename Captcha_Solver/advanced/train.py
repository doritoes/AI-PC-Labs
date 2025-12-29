import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os
import glob
from config import CHARS, CAPTCHA_LENGTH, WIDTH, HEIGHT, BATCH_SIZE, DEVICE

# --- 1. Early Stopping Helper ---
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# --- 2. Model Architecture (Same as before) ---
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

# --- 3. Dataset & Training ---
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
        label = torch.tensor([CHARS.find(c) for c in os.path.basename(path).split('.')[0]], dtype=torch.long)
        return image, label

def train():
    print(f"🚀 Arrow Lake XPU Training (Early Stopping Enabled)")
    model = CaptchaModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device=DEVICE)
    early_stopper = EarlyStopping(patience=3) # Stop if no gain for 3 epochs

    # Load and Split Data (80% Train, 20% Val)
    full_dataset = CaptchaDataset(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=True)

    for epoch in range(100): # High cap, Early Stopping will likely halt it sooner
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="xpu", dtype=torch.float16):
                loss = sum(criterion(out, labels[:, j]) for j, out in enumerate(model(images)))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with torch.amp.autocast(device_type="xpu", dtype=torch.float16):
                    loss = sum(criterion(out, labels[:, j]) for j, out in enumerate(model(images)))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")

        # Check Early Stopping
        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print("🛑 Early stopping triggered. Saving best model...")
            break

    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "..", "captcha_model.pth"))

if __name__ == "__main__":
    train()
