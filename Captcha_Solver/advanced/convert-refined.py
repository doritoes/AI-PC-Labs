import os
import torch
import openvino as ov
from model import AdvancedCaptchaModel
import config

def convert_to_ir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, "refined_advanced_model.pth")
    output_dir = os.path.join(current_dir, "openvino_models")
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    print(f"🔄 Converting Refined Weights to FP32 IR...")
    model = AdvancedCaptchaModel()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # Use 0.0 to 1.0 range (ToTensor default)
    example_input = torch.randn(1, 1, config.HEIGHT, config.WIDTH)
    ov_model = ov.convert_model(model, example_input=example_input)
    
    # Static shape for NPU performance
    ov_model.reshape({0: [1, 1, config.HEIGHT, config.WIDTH]})
    
    xml_path = os.path.join(output_dir, "refined_model_fp32.xml")
    ov.save_model(ov_model, xml_path)
    print(f"✅ Created FP32 Baseline: {xml_path}")

if __name__ == "__main__":
    convert_to_ir()
