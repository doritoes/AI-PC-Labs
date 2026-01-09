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
    output_dir = os.path.join(current_dir, "openvino_model_int8")
    
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

    # 3. Create Calibration Data
    print("🧪 Generating 300 samples for INT8 calibration...")
    generator = ImageCaptcha(width=config.WIDTH, height=config.HEIGHT)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # FIX: NNCF for PyTorch models needs the transform_fn to return a TENSOR
    # not a numpy array. The numpy conversion happens inside OpenVINO later.
    def transform_fn(data_item):
        return data_item.unsqueeze(0) # Keep as torch.Tensor

    calibration_images = []
    for _ in range(300):
        text = ''.join(np.random.choice(list(config.CHARS), config.CAPTCHA_LENGTH))
        img = generator.generate_image(text).convert('L')
        calibration_images.append(transform(img))
    
    calibration_dataset = nncf.Dataset(calibration_images, transform_fn)

    # 4. Quantize to INT8
    print("🚀 Quantizing weights to INT8...")
    # This will now pass a Tensor to conv2d, fixing the TypeError
    quantized_model = nncf.quantize(model, calibration_dataset)

    # 5. Export to OpenVINO IR
    print("💾 Exporting to OpenVINO IR format...")
    example_input = torch.randn(1, 1, config.HEIGHT, config.WIDTH)
    ov_model = ov.convert_model(quantized_model, example_input=example_input)
    
    # Force static shape for NPU
    ov_model.reshape({0: [1, 1, config.HEIGHT, config.WIDTH]})

    # 6. Save
    xml_path = os.path.join(output_dir, "captcha_model_int8.xml")
    ov.save_model(ov_model, xml_path)

    print("-" * 50)
    print("✅ CONVERSION SUCCESSFUL")
    print(f"📍 Location: {output_dir}")
    print("-" * 50)

if __name__ == "__main__":
    convert_and_quantize()
