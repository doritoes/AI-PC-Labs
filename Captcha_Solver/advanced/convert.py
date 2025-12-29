import torch
import os
import openvino as ov
import numpy as np
from train import CaptchaModel
from config import WIDTH, HEIGHT, DEVICE, CAPTCHA_LENGTH

def convert():
    # 1. Setup Paths
    # Finds the root directory (Captcha-AI) relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    model_path = os.path.join(root_dir, "captcha_model.pth")
    ov_model_dir = os.path.join(root_dir, "openvino_model")
    
    if not os.path.exists(ov_model_dir):
        os.makedirs(ov_model_dir)

    # 2. Check if the PyTorch model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Could not find '{model_path}'.")
        print("Please run train.py first to generate the model weights.")
        return

    # 3. Load the PyTorch Model
    print(f"🔄 Loading PyTorch weights from: {model_path}")
    model = CaptchaModel()
    # We load to CPU for the conversion process
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 4. Define Input Shape
    # Format: [Batch, Channels, Height, Width]
    # Based on your config, typically [1, 1, 80, 200]
    dummy_input = torch.randn(1, 1, HEIGHT, WIDTH)

    # 5. Convert to OpenVINO Intermediate Representation (IR)
    print("🚀 Starting OpenVINO conversion (2025.4.1 Stack)...")
    try:
        # Convert the PyTorch model to an OpenVINO Model object
        ov_model = ov.convert_model(model, example_input=dummy_input)

        # 6. Save as FP16
        # Compressing to FP16 is crucial for maximum performance on the NPU
        ov_xml_path = os.path.join(ov_model_dir, "captcha_model.xml")
        ov.save_model(ov_model, ov_xml_path, compress_to_fp16=True)
        
        print("-" * 50)
        print(f"✅ Conversion Successful!")
        print(f"📁 Model Directory: {ov_model_dir}")
        print(f"📄 Topology File:  captcha_model.xml")
        print(f"📄 Weights File:   captcha_model.bin")
        print("-" * 50)
        print("💡 Target Device Recommendation:")
        print("   - For lowest power: Use 'NPU'")
        print("   - For highest throughput: Use 'GPU'")
        print("-" * 50)

    except Exception as e:
        print(f"❌ An error occurred during conversion: {e}")

if __name__ == "__main__":
    convert()
