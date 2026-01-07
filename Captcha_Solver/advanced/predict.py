import torch
from torchvision import transforms
from captcha.image import ImageCaptcha
import numpy as np
import string
import config
import os
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

def run_combined_test(num_examples=10, diagnostic_samples=100):
    device = torch.device(config.DEVICE)
    model = AdvancedCaptchaModel().to(device)
    
    if os.path.exists("advanced_lab_model.pth"):
        model.load_state_dict(torch.load("advanced_lab_model.pth", map_location=device))
    else:
        print("❌ Error: advanced_lab_model.pth not found!")
        return

    model.eval()
    generator = ImageCaptcha(width=config.WIDTH, height=config.HEIGHT)
    
    # --- PART 1: VISION TEST (Live Examples) ---
    print(f"\n--- Model Vision Test ({num_examples} Samples) ---")
    print(f"{'REAL TEXT':<15} | {'PREDICTED':<15} | {'RESULT'}")
    print("-" * 45)

    with torch.no_grad():
        for _ in range(num_examples):
            real_text = ''.join(np.random.choice(list(config.CHARS), config.CAPTCHA_LENGTH))
            img = generator.generate_image(real_text).convert('L')
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
            output = model(img_tensor)
            pred_text = decode_output(output)[0]
            
            status = "✅ MATCH" if real_text == pred_text else "❌ FAIL"
            print(f"{real_text:<15} | {pred_text:<15} | {status}")

    # --- PART 2: DIAGNOSTICS (Statistics) ---
    total_chars = 0
    correct_chars = 0
    full_matches = 0
    confusion_map = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for _ in range(diagnostic_samples):
            real_text = ''.join(np.random.choice(list(config.CHARS), config.CAPTCHA_LENGTH))
            img = generator.generate_image(real_text).convert('L')
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
            output = model(img_tensor)
            pred_text = decode_output(output)[0]
            
            if real_text == pred_text:
                full_matches += 1
            for r, p in zip(real_text, pred_text):
                total_chars += 1
                if r == p: correct_chars += 1
                else: confusion_map[r][p] += 1

    char_acc = (correct_chars / total_chars) * 100
    solve_rate = (full_matches / diagnostic_samples) * 100
    
    print(f"\n📊 SUMMARY STATISTICS ({diagnostic_samples} Samples):")
    print(f"{'Character-Level Accuracy:':<30} {char_acc:>6.2f}%")
    print(f"{'Full Captcha Solve Rate:':<30} {solve_rate:>6.2f}%")
    print("-" * 45)
    
    print("\n🔍 TOP 5 CONFUSIONS (Real -> Predicted):")
    confusions = [(r, p, count) for r, preds in confusion_map.items() for p, count in preds.items()]
    confusions.sort(key=lambda x: x[2], reverse=True)
    
    for real, pred, count in confusions[:5]:
        print(f"  '{real}' mistaken for '{pred}' ({count} times)")

if __name__ == "__main__":
    run_combined_test()
