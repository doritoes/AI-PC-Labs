"""
technically conver.py performed the INT8 quantization

However, it is best practice to have a standalone quantize.py that is separate from a simple "conversion" script. This allows you to re-run the optimization with different calibration settings (like more samples or different noise levels) without touching the main export logic.
"""
import os
import torch
import nncf
import openvino as ov
import numpy as np
import time
from captcha.image import ImageCaptcha
from torchvision import transforms
from model import AdvancedCaptchaModel
import config

def run_quantization():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, "advanced_lab_model.pth")
    output_dir = os.path.join(current_dir, "openvino_int8_model")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("🔄 Loading Advanced Alphanumeric Weights...")
    model = AdvancedCaptchaModel()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    print("🧪 Generating 1000 high-diversity calibration samples...")
    generator = ImageCaptcha(width=config.WIDTH, height=config.HEIGHT)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    calibration_images = []
    for i in range(1000):
        text = ''.join(np.random.choice(list(config.CHARS), config.CAPTCHA_LENGTH))
        img = generator.generate_image(text).convert('L')
        calibration_images.append(transform(img))
        if (i + 1) % 250 == 0:
            print(f"  > Ready: {i + 1}/1000")

    def transform_fn(data_item):
        return data_item.unsqueeze(0)

    calibration_dataset = nncf.Dataset(calibration_images, transform_fn)

    print("\n🚀 Running MIXED-Precision Quantization...")
    
    start_time = time.time()
    
    # MIXED: Allows asymmetric quantization for better OCR precision
    # fast_bias_correction: Must be True for the PyTorch-NNCF backend
    quantized_model = nncf.quantize(
        model, 
        calibration_dataset,
        preset=nncf.QuantizationPreset.MIXED,
        subset_size=1000,
        fast_bias_correction=True 
    )

    print(f"\n✅ Optimization Finished in {time.time() - start_time:.1f}s")

    print("💾 Saving Static INT8 model (200x80)...")
    example_input = torch.randn(1, 1, config.HEIGHT, config.WIDTH)
    ov_model = ov.convert_model(quantized_model, example_input=example_input)
    
    ov_model.reshape({0: [1, 1, config.HEIGHT, config.WIDTH]})

    xml_path = os.path.join(output_dir, "captcha_model_int8.xml")
    ov.save_model(ov_model, xml_path)

    print("-" * 50)
    print(f"🏁 DONE! Updated model saved: {xml_path}")
    print("-" * 50)

if __name__ == "__main__":
    run_quantization()
