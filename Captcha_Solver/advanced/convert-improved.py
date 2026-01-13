import os
import torch
import openvino as ov
from model import AdvancedCaptchaModel
import config

def convert_baseline():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, "advanced_lab_model.pth")
    
    if not os.path.exists(weights_path):
        print(f"❌ Error: {weights_path} not found!")
        return

    print("📂 Loading Baseline Advanced Model (60% Solve Rate)...")
    model = AdvancedCaptchaModel()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # Convert to FP32 (Full Precision)
    example_input = torch.randn(1, 1, 80, 200)
    ov_model = ov.convert_model(model, example_input=example_input)
    
    # Static shape for NPU performance
    ov_model.reshape({0: [1, 1, 80, 200]})
    
    ov.save_model(ov_model, "baseline_npu_fp32.xml")
    print("✅ Created baseline_npu_fp32.xml")

if __name__ == "__main__":
    convert_baseline()
