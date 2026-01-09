"""
tehcnically conver.py performed the INT8 quantization

However, it is best practice to have a standalone quantize.py that is separate from a simple "conversion" script. This allows you to re-run the optimization with different calibration settings (like more samples or different noise levels) without touching the main export logic.
"""
import os
import torch
import nncf
import openvino as ov
import numpy as np
from captcha.image import ImageCaptcha
from torchvision import transforms
from model import AdvancedCaptchaModel
import config

def run_quantization():
    # 1. Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, "advanced_lab_model.pth")
    output_dir = os.path.join(current_dir, "openvino_int8_model")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Load the Weights
    print("🔄 Loading Advanced Lab Model for INT8 Optimization...")
    model = AdvancedCaptchaModel()
    if not os.path.exists(weights_path):
        print(f"❌ Error: {weights_path} not found!")
        return
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # 3. Calibration Dataset
    # We use 500 samples here instead of 300 for a slightly better "fine-tune"
    print("🧪 Generating 500 fresh samples for NPU calibration...")
    generator = ImageCaptcha(width=config.WIDTH, height=config.HEIGHT)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    def transform_fn(data_item):
        # Keep as Tensor for NNCF-Torch compatibility
        return data_item.unsqueeze(0)

    calibration_images = []
    for _ in range(500):
        # Calibrate using the full range of possible alphanumeric characters
        text = ''.join(np.random.choice(list(config.CHARS), config.CAPTCHA_LENGTH))
        img = generator.generate_image(text).convert('L')
        calibration_images.append(transform(img))
    
    calibration_dataset = nncf.Dataset(calibration_images, transform_fn)

    # 4. Execute Quantization
    print("🚀 Running NNCF INT8 Quantization...")
    # This aligns the weights to 8-bit integers (INT8) for the NPU
    quantized_model = nncf.quantize(
        model, 
        calibration_dataset,
        preset=nncf.QuantizationPreset.PERFORMANCE # Prioritizes speed for the game script
    )

    # 5. Export to OpenVINO IR
    print("💾 Saving Static INT8 model for Intel AI Boost...")
    example_input = torch.randn(1, 1, config.HEIGHT, config.WIDTH)
    ov_model = ov.convert_model(quantized_model, example_input=example_input)
    
    # Critical: Set static shape metadata so the NPU doesn't have to recompile
    ov_model.reshape({0: [1, 1, config.HEIGHT, config.WIDTH]})

    xml_path = os.path.join(output_dir, "captcha_model_int8.xml")
    ov.save_model(ov_model, xml_path)

    print("-" * 50)
    print("✅ INT8 QUANTIZATION COMPLETE")
    print(f"📂 Model saved in: {output_dir}")
    print(f"⚙️ Preset: PERFORMANCE")
    print("-" * 50)

if __name__ == "__main__":
    run_quantization()
