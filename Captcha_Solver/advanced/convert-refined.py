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

def convert_refined_to_npu():
    # 1. Setup Paths (Same output folder, different filenames)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, "refined_advanced_model.pth")
    output_dir = os.path.join(current_dir, "openvino_int8_model")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Load the Refined Model
    print(f"📂 Loading Refined Model: {weights_path}")
    model = AdvancedCaptchaModel()
    if not os.path.exists(weights_path):
        print(f"❌ Error: {weights_path} not found! Run refine.py first.")
        return
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # 3. Create Calibration Data (Increased to 1500 for the refined boundaries)
    print("🧪 Generating 1500 high-diversity samples for calibration...")
    generator = ImageCaptcha(width=config.WIDTH, height=config.HEIGHT)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    calibration_images = []
    for i in range(1500):
        text = ''.join(np.random.choice(list(config.CHARS), config.CAPTCHA_LENGTH))
        img = generator.generate_image(text).convert('L')
        calibration_images.append(transform(img))
        if (i + 1) % 500 == 0:
            print(f"  > Prepared {i+1}/1500 samples...")

    def transform_fn(data_item):
        return data_item.unsqueeze(0)

    calibration_dataset = nncf.Dataset(calibration_images, transform_fn)

    # 4. Define the Accuracy Shield
    # Protects the refined classification layers from precision loss
    ignored_scope = nncf.IgnoredScope(
        patterns=[".*fc.*", ".*output.*"]
    )

    # 5. Execute Quantization
    print("🚀 Starting Mixed-Precision Quantization (Refined)...")
    start_time = time.time()
    
    try:
        quantized_model = nncf.quantize(
            model, 
            calibration_dataset,
            preset=nncf.QuantizationPreset.MIXED,
            subset_size=1500,
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
            subset_size=1500,
            fast_bias_correction=True
        )

    # 6. Export to OpenVINO IR
    print("💾 Exporting to OpenVINO IR for Intel NPU...")
    example_input = torch.randn(1, 1, config.HEIGHT, config.WIDTH)
    ov_model = ov.convert_model(quantized_model, example_input=example_input)

    # Static Reshape for NPU peak performance
    ov_model.reshape({0: [1, 1, config.HEIGHT, config.WIDTH]})

    # 7. Save Model Files with UNIQUE NAMES
    xml_path = os.path.join(output_dir, "refined_model_int8.xml")
    ov.save_model(ov_model, xml_path)

    print("-" * 50)
    print("✅ REFINED CONVERSION SUCCESSFUL")
    print(f"📍 Model: {xml_path}")
    print(f"⏱️ Time: {time.time() - start_time:.1f}s")
    print("-" * 50)

if __name__ == "__main__":
    convert_refined_to_npu()
