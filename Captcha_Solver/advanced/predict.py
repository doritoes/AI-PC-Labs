import torch
import torch.nn.functional as F
from torchvision import transforms
from captcha.image import ImageCaptcha
import numpy as np
import string
import config
import os
from model import AdvancedCaptchaModel
from collections import defaultdict

def decode_with_confidence(outputs):
    out_reshaped = outputs.view(-1, 6, 62)
    probs = F.softmax(out_reshaped, dim=2)
    confidences, indices = torch.max(probs, dim=2)
    
    results = []
    avg_confs = []
    for i in range(len(indices)):
        chars = [config.CHARS[idx] for idx in indices[i]]
        results.append("".join(chars))
        avg_confs.append(confidences[i].tolist())
    return results, avg_confs

def run_complete_diagnostic(num_examples=10, diagnostic_samples=100):
    device = torch.device(config.DEVICE)
    model = AdvancedCaptchaModel().to(device)
    
    if os.path.exists("advanced_lab_model.pth"):
        model.load_state_dict(torch.load("advanced_lab_model.pth", map_location=device))
    else:
        print("❌ Error: No model file found.")
        return

    model.eval()
    generator = ImageCaptcha(width=config.WIDTH, height=config.HEIGHT)
    
    # --- PART 1: VISION TEST (Live Examples) ---
    print(f"\n--- Model Vision Test with Confidence ---")
    print(f"{'REAL':<8} | {'PRED':<8} | {'CONFIDENCE (Per Char)':<20} | {'AVG'}  | {'RESULT'}")
    print("-" * 75)

    with torch.no_grad():
        for _ in range(num_examples):
            real_text = ''.join(np.random.choice(list(config.CHARS), config.CAPTCHA_LENGTH))
            img = generator.generate_image(real_text).convert('L')
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
            
            output = model(img_tensor)
            preds, confs = decode_with_confidence(output)
            
            pred_text = preds[0]
            char_confs = confs[0]
            avg_conf = sum(char_confs) / 6
            conf_str = " ".join([f"{c:.2f}"[1:] for c in char_confs]) 
            
            status = "✅ MATCH" if real_text == pred_text else "❌ FAIL"
            print(f"{real_text:<8} | {pred_text:<8} | {conf_str:<20} | {avg_conf:>4.0%} | {status}")

    # --- PART 2: DIAGNOSTICS (Statistics) ---
    total_chars = 0
    correct_chars = 0
    full_matches = 0
    confusion_map = defaultdict(lambda: defaultdict(int))

    print(f"\n📊 Running Statistical Analysis ({diagnostic_samples} samples)...")
    with torch.no_grad():
        for _ in range(diagnostic_samples):
            real_text = ''.join(np.random.choice(list(config.CHARS), config.CAPTCHA_LENGTH))
            img = generator.generate_image(real_text).convert('L')
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
            output = model(img_tensor)
            preds, _ = decode_with_confidence(output)
            pred_text = preds[0]
            
            if real_text == pred_text:
                full_matches += 1
            for r, p in zip(real_text, pred_text):
                total_chars += 1
                if r == p: 
                    correct_chars += 1
                else: 
                    confusion_map[r][p] += 1

    char_acc = (correct_chars / total_chars) * 100
    solve_rate = (full_matches / diagnostic_samples) * 100
    
    print(f"\n📈 FINAL STATS:")
    print(f"{'Character-Level Accuracy:':<30} {char_acc:>6.2f}%")
    print(f"{'Full Captcha Solve Rate:':<30} {solve_rate:>6.2f}%")
    print("-" * 45)
    
    print("\n🔍 TOP 5 CONFUSIONS (Real -> Predicted):")
    # Cleaned up the list comprehension that caused the error
    confusions = []
    for r, preds in confusion_map.items():
        for p, count in preds.items():
            confusions.append((r, p, count))
            
    confusions.sort(key=lambda x: x[2], reverse=True)
    for real, pred, count in confusions[:5]:
        print(f"  '{real}' mistaken for '{pred}' ({count} times)")

if __name__ == "__main__":
    run_complete_diagnostic()
