"""
Shows progress of the model advanced_lab_model.pth
"""
import torch
from torchvision import transforms
from captcha.image import ImageCaptcha
import numpy as np
import string
import config
from model import AdvancedCaptchaModel
from PIL import Image

def decode_output(outputs):
    """Converts the model's raw numbers back into a 6-character string."""
    out_reshaped = outputs.view(-1, 6, 62)
    indices = torch.argmax(out_reshaped, dim=2)
    
    results = []
    for batch in indices:
        chars = [config.CHARS[i] for i in batch]
        results.append("".join(chars))
    return results

def test_model(num_samples=5):
    device = torch.device(config.DEVICE)
    model = AdvancedCaptchaModel().to(device)
    
    # Load your hard-earned weights
    if not torch.cuda.is_available() and config.DEVICE == "xpu":
        # Handle XPU loading logic
        model.load_state_dict(torch.load("advanced_lab_model.pth", map_location=device))
    else:
        model.load_state_dict(torch.load("advanced_lab_model.pth"))
    
    model.eval() # Set to evaluation mode
    generator = ImageCaptcha(width=config.WIDTH, height=config.HEIGHT)
    
    print(f"\n--- Model Vision Test ({num_samples} Samples) ---")
    print(f"{'REAL TEXT':<15} | {'PREDICTED':<15} | {'RESULT'}")
    print("-" * 45)

    with torch.no_grad():
        for _ in range(num_samples):
            # Generate a fresh captcha the model has NEVER seen
            real_text = ''.join(np.random.choice(list(config.CHARS), config.CAPTCHA_LENGTH))
            img = generator.generate_image(real_text).convert('L')
            
            # Prepare image for model
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
            
            # Get prediction
            output = model(img_tensor)
            predicted_text = decode_output(output)[0]
            
            status = "✅ MATCH" if real_text == predicted_text else "❌ FAIL"
            print(f"{real_text:<15} | {predicted_text:<15} | {status}")

if __name__ == "__main__":
    test_model(10) # Test 10 random samples
