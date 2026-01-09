import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from captcha.image import ImageCaptcha
from model import AdvancedCaptchaModel
import config

# --- 1. THE HYBRID DATASET ENGINE ---
class HybridCaptchaDataset(Dataset):
    def __init__(self, size, phase="digits", transform=None):
        self.size = size
        self.transform = transform
        self.phase = phase
        self.generator = ImageCaptcha(width=config.WIDTH, height=config.HEIGHT)
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Determine character pool based on phase
        if self.phase == "digits":
            pool = "0123456789"
        else:
            pool = config.CHARS
            
        target = "".join([np.random.choice(list(pool)) for _ in range(config.CAPTCHA_LENGTH)])
        
        # Color is enabled for visual depth, then converted to Grayscale if needed by model
        img = self.generator.generate_image(target)
        
        if self.transform:
            img = self.transform(img)
            
        # Encode label
        label = torch.LongTensor([config.CHARS.find(c) for c in target])
        return img, label

# --- 2. THE TRANSFORM PIPELINE ---
# No blurring as requested. Focus on Perspective and Geometry.
def get_transforms(phase):
    t_list = [transforms.Grayscale()]
    
    if phase == "refinement":
        # Perspective is the ultimate 'Exam' preparation
        t_list.append(transforms.RandomPerspective(distortion_scale=0.2, p=0.7))
        t_list.append(transforms.RandomRotation(degrees=12))
    elif phase == "alphanumeric":
        t_list.append(transforms.RandomRotation(degrees=8))
        
    t_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transforms.Compose(t_list)

# --- 3. TRAINING ORCHESTRATOR ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdvancedCaptchaModel().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Phase Management
    phases = [
        {"name": "digits", "epochs": (1, 31), "lr": 0.001},
        {"name": "alphanumeric", "epochs": (31, 71), "lr": 0.0005},
        {"name": "refinement", "epochs": (71, 101), "lr": 0.0001}
    ]

    for p in phases:
        print(f"\n🚀 STARTING PHASE: {p['name'].upper()} (LR: {p['lr']})")
        optimizer = optim.Adam(model.parameters(), lr=p['lr'])
        
        # Fresh image heavy mix (80% fresh in refinement)
        dataset = HybridCaptchaDataset(size=20000, phase=p['name'], transform=get_transforms(p['name']))
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(p['epochs'][0], p['epochs'][1]):
            model.train()
            total_loss = 0
            
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                outputs = model(imgs).view(-1, config.CAPTCHA_LENGTH, len(config.CHARS))
                
                loss = 0
                for i in range(config.CAPTCHA_LENGTH):
                    loss += criterion(outputs[:, i, :], labels[:, i])
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "advanced_lab_model.pth")
    print("\n✅ 100 Epoch Marathon Complete.")

if __name__ == "__main__":
    train()
