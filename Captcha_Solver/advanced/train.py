import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import gc
from tqdm import tqdm
from captcha.image import ImageCaptcha
from model import AdvancedCaptchaModel
import config

class HybridCaptchaDataset(Dataset):
    def __init__(self, size, phase="digits", transform=None):
        self.size = size
        self.transform = transform
        self.phase = phase
        self.generator = ImageCaptcha(width=config.WIDTH, height=config.HEIGHT)
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        pool = "0123456789" if self.phase == "digits" else config.CHARS
        target = "".join([np.random.choice(list(pool)) for _ in range(config.CAPTCHA_LENGTH)])
        
        # Fresh generation happens here - CPU intensive
        img = self.generator.generate_image(target).convert('L')
        
        if self.transform:
            img = self.transform(img)
            
        label = torch.LongTensor([config.CHARS.find(c) for c in target])
        return img, label

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdvancedCaptchaModel().to(device)
    criterion = nn.CrossEntropyLoss()
    
    phases = [
        {"name": "digits", "epochs": (1, 31), "lr": 0.001},
        {"name": "alphanumeric", "epochs": (31, 71), "lr": 0.0005},
        {"name": "refinement", "epochs": (71, 101), "lr": 0.0001}
    ]

    for p in phases:
        print(f"\n🚀 PHASE: {p['name'].upper()} | Target: {p['epochs'][1]-p['epochs'][0]} Epochs")
        optimizer = optim.Adam(model.parameters(), lr=p['lr'])
        
        # Reduced size to 10k to prevent RAM bloat on 16GB systems
        dataset = HybridCaptchaDataset(size=10000, phase=p['name'], transform=get_transforms(p['name']))
        # num_workers=0 is safer for debugging "frozen" scripts
        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

        for epoch in range(p['epochs'][0], p['epochs'][1]):
            model.train()
            total_loss = 0
            
            # This progress bar will tell you EXACTLY if it's moving
            pbar = tqdm(loader, desc=f"Epoch {epoch}", unit="batch")
            
            for imgs, labels in pbar:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                outputs = model(imgs).view(-1, config.CAPTCHA_LENGTH, len(config.CHARS))
                
                loss = 0
                for i in range(config.CAPTCHA_LENGTH):
                    loss += criterion(outputs[:, i, :], labels[:, i])
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item()/config.CAPTCHA_LENGTH:.4f}")

            # Force memory cleanup after every epoch
            gc.collect()
            torch.save(model.state_dict(), "advanced_lab_model.pth")

# Placeholder for the transform function mentioned in previous turns
def get_transforms(phase):
    t_list = [transforms.Grayscale()]
    if phase == "refinement":
        t_list.append(transforms.RandomPerspective(distortion_scale=0.2, p=0.7))
        t_list.append(transforms.RandomRotation(degrees=12))
    t_list.extend([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return transforms.Compose(t_list)

if __name__ == "__main__":
    train()
