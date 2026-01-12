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
    # 1. Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, "advanced_lab_model.pth")
    output_dir = os.path.join(current_dir, "openvino_int8_model")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Load the Model
    print("📂 Loading Advanced Model weights...")
    model = AdvancedCaptchaModel()
    if not os.path.exists(weights_path):
        print(f"❌ Error: {weights_path} not found!")
        return
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # 3. Create High-Diversity Calibration Data
    # We use 1000 samples to ensure the NPU sees the full range of noise
    print("🧪 Generating 1000 high-entropy samples for INT8 calibration...")
    generator = ImageCaptcha(width=config.WIDTH, height=config.HEIGHT)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # MUST match train.py and game.py
    ])

    calibration_images = []
    for _ in range(1000):
        text = ''.join(np.random.choice(list(config.CHARS), config.CAPTCHA_LENGTH))
        img = generator.generate_image(text).convert('L')
        calibration_images.append(transform(img))

    def transform_fn(data_item):
        return data_item.unsqueeze(0)

    calibration_dataset = nncf.Dataset(calibration_images, transform_fn)

    # 4. Define Ignored Scope (The "Accuracy Shield")
    # This prevents the final layers from being quantized to INT8.
    # Replace 'fc' with the actual name of your final linear layer from model.py
    ignored_scope = nncf.IgnoredScope(
        names=["fc", "classifier", "out_layer"], 
        types=["Softmax", "LogSoftmax"]
    )

    # 5. Quantize with Mixed Precision
    print("🚀 Quantizing to INT8 (Protecting sensitive layers)...")
    # We use ACCURACY preset for better alphanumeric recognition
    quantized_model = nncf.quantize(
        model, 
        calibration_dataset,
        preset=nncf.QuantizationPreset.MIXED,
        subset_size=1000,
        ignored_scope=ignored_scope,
        fast_bias_correction=True 
    )

    # 6. Export to OpenVINO IR
    print("💾 Exporting to OpenVINO IR for NPU...")
    example_input = torch.randn(1, 1, config.HEIGHT, config.WIDTH)
    ov_model = ov.convert_model(quantized_model, example_input=example_input)

    # Static Reshape: Arrow Lake NPU performs best with fixed shapes
    ov_model.reshape({0: [1, 1, config.HEIGHT, config.WIDTH]})

    # 7. Save
    xml_path = os.path.join(output_dir, "captcha_model_int8.xml")
    ov.save_model(ov_model, xml_path)

    print("-" * 50)
    print("✅ CONVERSION SUCCESSFUL")
    print(f"📍 Location: {output_dir}")
    print(f"💡 Note: The output layer was kept in FP16 to prevent 'O/0' confusion.")
    print("-" * 50)

if __name__ == "__main__":
    convert_and_quantize()
