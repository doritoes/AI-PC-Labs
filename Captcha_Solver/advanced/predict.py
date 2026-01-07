import torch
from torchvision import transforms
from captcha.image import ImageCaptcha
import numpy as np
import string
import config
from model import AdvancedCaptchaModel
from collections import defaultdict

def decode_output(outputs):
    out_reshaped = outputs.view(-1, 6, 62)
    indices = torch.argmax(out_reshaped, dim=2)
    results = []
    for batch in indices:
        chars = [config.CHARS[i] for i in batch]
        results.append("".join(chars))
    return results

def test_diagnostic(num_samples=50):
    device = torch.device(config.DEVICE)
    model = AdvancedCaptchaModel().to(device)
    
    if os.path.exists("advanced_lab_model.pth"):
        model.load_state_dict(torch.load("advanced_lab_model.pth", map_location=device))
    
    model.eval()
    generator = ImageCaptcha(width=config.WIDTH, height=config.HEIGHT)
    
    total_chars = 0
    correct_chars = 0
    full_matches = 0
    confusion_map = defaultdict(lambda: defaultdict(int))

    print(f"\n--- Model Diagnostic ({num_samples} Samples) ---")
    
    with torch.no_grad():
        for _ in range(num_samples):
            real_text = ''.join(np.random.choice(list(config.CHARS), config.CAPTCHA_LENGTH))
            img = generator.generate_image(real_text).convert('L')
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
            
            output = model(img_tensor)
            pred_text = decode_output(output)[0]
            
            if real_text == pred_text:
                full_matches += 1
            
            for r, p in zip(real_text, pred_text):
                total_chars += 1
                if r == p:
                    correct_chars += 1
                else:
                    confusion_map[r][p] += 1

    # --- REPORTING ---
    char_acc = (correct_chars / total_chars) * 100
    solve_rate = (full_matches / num_samples) * 100
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"{'Character-Level Accuracy:':<30} {char_acc:>6.2f}%")
    print(f"{'Full Captcha Solve Rate:':<30} {solve_rate:>6.2f}%")
    print("-" * 45)
    
    print("\n🔍 TOP 5 CONFUSIONS (Real -> Predicted):")
    # Flatten and sort the confusion map
    confusions = []
    for real, preds in confusion_map.items():
        for pred, count in preds.items():
            confusions.append((real, pred, count))
    
    confusions.sort(key=lambda x: x[2], reverse=True)
    
    for real, pred, count in confusions[:5]:
        print(f"  '{real}' mistaken for '{pred}' ({count} times)")

if __name__ == "__main__":
    import os
    test_diagnostic(100) # Increased sample size for better stats
