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

def run_refined_quantization():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Load the specific weights from your successful refinement run
    weights_path = os.path.join(current_dir, "refined_advanced_model.pth")
    output_dir = os.path.join(current_dir, "openvino_int8_model")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"🔄 Loading Refined Alphanumeric Weights: {weights_path}")
    model = AdvancedCaptchaModel()
    if not os.path.exists(weights_path):
        print(f"❌ Error: {weights_path} not found! Run refine.py first.")
        return
    
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # 1. High-Entropy Calibration (Match your refinement diversity)
    print("🧪 Generating 1500 high-diversity calibration samples...")
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
            print(f"  > Ready: {i + 1}/1500")

    def transform_fn(data_item):
        return data_item.unsqueeze(0)

    calibration_dataset = nncf.Dataset(calibration_images, transform_fn)

    # 2. Accuracy Shield (Crucial to protect the refined decision layers)
    ignored_scope = nncf.IgnoredScope(
        patterns=[".*fc.*", ".*output.*"]
    )

    print("\n🚀 Running MIXED-Precision Quantization on Refined Model...")
    start_time = time.time()
    
    # MIXED preset ensures NPU speed for Convs and FP16 for Classification
    quantized_model = nncf.quantize(
        model, 
        calibration_dataset,
        preset=nncf.QuantizationPreset.MIXED,
        subset_size=1500,
        ignored_scope=ignored_scope,
        fast_bias_correction=True 
    )

    print(f"\n✅ Optimization Finished in {time.time() - start_time:.1f}s")

    # 3. Export to OpenVINO IR
    print("💾 Saving Refined Static INT8 model (200x80) for NPU...")
    example_input = torch.randn(1, 1, config.HEIGHT, config.WIDTH)
    ov_model = ov.convert_model(quantized_model, example_input=example_input)
    
    # Lock shape for sub-10ms NPU latency
    ov_model.reshape({0: [1, 1, config.HEIGHT, config.WIDTH]})

    # Use 'refined_model_int8.xml' to avoid conflict with baseline
    xml_path = os.path.join(output_dir, "refined_model_int8.xml")
    ov.save_model(ov_model, xml_path)

    print("-" * 50)
    print(f"🏁 DONE! Refined NPU model saved: {xml_path}")
    print("-" * 50)

if __name__ == "__main__":
    run_refined_quantization()
