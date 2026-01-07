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
    """Returns the predicted string AND the confidence (probability) for each char."""
    out_reshaped = outputs.view(-1, 6, 62)
    # Apply Softmax to get probabilities (0.0 to 1.0)
    probs = F.softmax(out_reshaped, dim=2)
    
    confidences, indices = torch.max(probs, dim=2)
    
    results = []
    avg_confs = []
    for i in range(len(indices)):
        chars = [config.CHARS[idx] for idx in indices[i]]
        results.append("".join(chars))
        avg_confs.append(confidences[i].tolist()) # List of 6 confidences
        
    return results, avg_confs

def run_confidence_test(num_examples=10):
    device = torch.device(config.DEVICE)
    model = AdvancedCaptchaModel().to(device)
    
    if os.path.exists("advanced_lab_model.pth"):
        model.load_state_dict(torch.load("advanced_lab_model.pth", map_location=device))
    else:
        print("❌ Error: No model file found.")
        return

    model.eval()
    generator = ImageCaptcha(width=config.WIDTH, height=config.HEIGHT)
    
    print(f"\n--- Model Vision Test with Confidence ---")
    print(f"{'REAL':<8} | {'PRED':<8} | {'CONFIDENCE (Per Char)':<25} | {'AVG'}")
    print("-" * 70)

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
            
            # Format confidence string (e.g., .99 .45 .88...)
            conf_str = " ".join([f"{c:.2f}"[1:] for c in char_confs]) 
            
            status = "✅" if real_text == pred_text else "❌"
            print(f"{real_text:<8} | {pred_text:<8} | {conf_str:<25} | {avg_conf:.1%}")

if __name__ == "__main__":
    run_confidence_test()
