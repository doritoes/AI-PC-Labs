import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os, random, glob, time, gc, string

# Import from your config.py
from config import CHARS, CAPTCHA_LENGTH, WIDTH, HEIGHT, BATCH_SIZE, DEVICE, DATASET_SIZE

# [Same CaptchaModel and CaptchaDataset classes as before...]

def train():
    # Use the device string directly from your config
    device = torch.device("xpu") 
    model = CaptchaModel().to(device)
    
    # Standard Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    criterion = nn.CrossEntropyLoss()
    
    # Load dataset
    dataset_path = os.path.join(os.getcwd(), "dataset")
    full_ds = CaptchaDataset(dataset_path)
    train_size = int(0.9 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [train_size, len(full_ds)-train_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print(f"🚀 Starting Training on {torch.xpu.get_device_name(0)}")
    best_acc = 0.0

    for epoch in range(35): 
        epoch_start = time.time()
        model.train()
        total_loss = 0
        
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            
            # Use native XPU autocast for 16-bit precision
            with torch.amp.autocast(device_type="xpu", dtype=torch.float16):
                outputs = model(imgs)
                loss = criterion(outputs.view(-1, len(CHARS)), lbls.view(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation logic...
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

        # Smart Stop
        if val_acc >= 98.5:
            print("🎯 Target reached. Stopping.")
            break

        # Cleanup for the 16GB RAM limit
        torch.xpu.empty_cache()
        gc.collect()
        time.sleep(10) 

    torch.save(model.state_dict(), "captcha_model_final.pth")

if __name__ == "__main__":
    train()
