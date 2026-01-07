"""
Benchmark the advanced_lab_model.pth performance and speed.
"""
import time
import torch
import numpy as np
from torchvision import transforms
from captcha.image import ImageCaptcha
import config
from model import AdvancedCaptchaModel
import os

def benchmark(num_samples=1000):
    device = torch.device(config.DEVICE)
    model = AdvancedCaptchaModel().to(device)
    
    # Load Weights
    if os.path.exists("advanced_lab_model.pth"):
        try:
            model.load_state_dict(torch.load("advanced_lab_model.pth", map_location=device))
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            return
    else:
        print("❌ Error: advanced_lab_model.pth not found.")
        return

    model.eval()
    generator = ImageCaptcha(width=config.WIDTH, height=config.HEIGHT)
    
    print(f"🚀 Starting Benchmark: {num_samples} samples on {config.DEVICE}...")
    
    correct_full = 0
    start_time = time.time()

    with torch.no_grad():
        for i in range(num_samples):
            # 1. Generate
            real_text = ''.join(np.random.choice(list(config.CHARS), config.CAPTCHA_LENGTH))
            img = generator.generate_image(real_text).convert('L')
            
            # 2. Pre-process
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
            
            # 3. Predict
            output = model(img_tensor)
            
            # 4. Decode
            out_reshaped = output.view(-1, 6, 62)
            indices = torch.argmax(out_reshaped, dim=2)
            pred_text = "".join([config.CHARS[idx] for idx in indices[0]])

            if real_text == pred_text:
                correct_full += 1
            
            if (i + 1) % 100 == 0:
                print(f"Processed: {i + 1}/{num_samples}...")

    end_time = time.time()
    total_time = end_time - start_time
    
    # Metrics calculation
    solve_rate = (correct_full / num_samples) * 100
    avg_latency = (total_time / num_samples) * 1000 # in ms
    spm = (correct_full / total_time) * 60 if total_time > 0 else 0

    print("\n" + "="*30)
    print("🏁 BENCHMARK COMPLETE")
    print("="*30)
    print(f"{'Total Samples:':<20} {num_samples}")
    print(f"{'Total Time:':<20} {total_time:.2f}s")
    print(f"{'Solve Rate:':<20} {solve_rate:.2f}%")
    print(f"{'Avg Latency:':<20} {avg_latency:.2f}ms / captcha")
    print(f"{'Solves Per Min:':<20} {spm:.2f} SPM")
    print("="*30)

if __name__ == "__main__":
    benchmark(1000)
