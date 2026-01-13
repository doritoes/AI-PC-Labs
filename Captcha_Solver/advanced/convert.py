"""
convert PyTorch model to OpenVINO
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

def convert_and_quantize():
    """ convert and quantize model """
    # 1. Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, "advanced_lab_model.pth")
    output_dir = os.path.join(current_dir, "openvino_int8_model")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Load the Advanced Model
    print("📂 Loading Advanced Model weights...")
    model = AdvancedCaptchaModel()
    if not os.path.exists(weights_path):
        print(f"❌ Error: {weights_path} not found!")
        return
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # 3. Create Calibration Data (1000 samples)
    print("🧪 Generating 1000 high-diversity samples for calibration...")
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
            print(f"  > Prepared {i+1}/1000 samples...")

    def transform_fn(data_item):
        return data_item.unsqueeze(0)

    calibration_dataset = nncf.Dataset(calibration_images, transform_fn)

    # 4. Define the Accuracy Shield
    # Regex patterns to catch 'fc' and 'output' layers for high-precision
    ignored_scope = nncf.IgnoredScope(
        patterns=[".*fc.*", ".*output.*"]
    )

    # 5. Execute Quantization
    print("🚀 Starting Mixed-Precision Quantization...")

    try:
        # Attempt quantization with the precision shield
        quantized_model = nncf.quantize(
            model,
            calibration_dataset,
            preset=nncf.QuantizationPreset.MIXED,
            subset_size=1000,
            ignored_scope=ignored_scope,
            fast_bias_correction=True
        )
    except Exception as e:
        print(f"⚠️ Shielding failed: {e}")
        print("🔄 Falling back to Automatic Mixed-Precision...")
        quantized_model = nncf.quantize(
            model,
            calibration_dataset,
            preset=nncf.QuantizationPreset.MIXED,
            subset_size=1000,
            fast_bias_correction=True
        )

    # 6. Export to OpenVINO IR
    print("💾 Exporting to OpenVINO IR for Intel NPU...")
    example_input = torch.randn(1, 1, config.HEIGHT, config.WIDTH)
    ov_model = ov.convert_model(quantized_model, example_input=example_input)

    # Static Reshape for NPU peak performance
    ov_model.reshape({0: [1, 1, config.HEIGHT, config.WIDTH]})

    # 7. Save Model Files
    xml_path = os.path.join(output_dir, "captcha_model_int8.xml")
    ov.save_model(ov_model, xml_path)

    print("-" * 50)
    print("✅ CONVERSION SUCCESSFUL")
    print(f"📍 Model: {xml_path}")
    print("-" * 50)

if __name__ == "__main__":
    convert_and_quantize()
