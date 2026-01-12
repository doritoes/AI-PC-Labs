import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from captcha.image import ImageCaptcha
from torchvision import transforms
import numpy as np
from model import AdvancedCaptchaModel
import config
import os

# Custom Dataset that focuses on "Hard" characters
class HardCaptchaDataset(Dataset):
    def __init__(self, count=5000):
        self.count = count
        self.generator = ImageCaptcha(width=config.WIDTH, height=config.HEIGHT)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        # Focus on characters the NPU keeps missing
        self.hard_chars = "0OoQpP1ilILsS5Z2" 

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        # 70% of the time, pick from the hard list; 30% from the full set
        source = self.hard_chars if np.random.rand() < 0.7 else config.CHARS
        text = ''.join(np.random.choice(list(source), config.CAPTCHA_LENGTH))
        
        img = self.generator.generate_image(text).convert('L')
        img_t = self.transform(img)
        
        # Encode labels
        target = torch.zeros(config.CAPTCHA_LENGTH, len(config.CHARS))
        for i, char in enumerate(text):
            target[i, config.CHARS.find(char)] = 1
        return img_t, target.flatten()

def refine_model():
    device = torch.device("cpu") # Use "cuda" if you have an NVIDIA GPU
    model = AdvancedCaptchaModel().to(device)
    
    weights_path = "advanced_lab_model.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("📈 Loaded existing weights for refinement...")

    criterion = nn.MultiLabelSoftMarginLoss()
    # Very low learning rate to avoid "Catastrophic Forgetting"
    optimizer = optim.Adam(model.parameters(), lr=1e-5) 

    dataset = HardCaptchaDataset(count=10000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    print("🚀 Starting Hard-Example Refinement...")

    for epoch in range(5): # Short burst of refinement
        total_loss = 0
        for i, (imgs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/5], Step [{i+1}/313], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "refined_advanced_model.pth")
    print("✅ Refinement complete. Saved as: refined_advanced_model.pth")

if __name__ == "__main__":
    refine_model()
