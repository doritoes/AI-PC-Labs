"""
quantize the OpenVINO model to INT8
"""
import os
import time
import torch
import nncf
import openvino as ov
import numpy as np
from captcha.image import ImageCaptcha
from torchvision import transforms
from model import AdvancedCaptchaModel
import config

def run_quantization():
    """ quantize model to INT8 """
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

    # We use the exact same normalization as the game loop
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    calibration_images = []
    for i in range(1000):
        # Generate random text from the full 62-char set
        text = ''.join(np.random.choice(list(config.CHARS), config.CAPTCHA_LENGTH))
        img = generator.generate_image(text).convert('L')

        # Optimization: Occasionally add a tiny bit of Gaussian noise to calibration
        # to help the INT8 weights handle the "grain" in the game's CAPTCHAs
        img_t = transform(img)
        if i % 10 == 0:
            img_t = img_t + torch.randn_like(img_t) * 0.01

        calibration_images.append(img_t)
        if (i + 1) % 250 == 0:
            print(f"  > Ready: {i + 1}/1000")

    def transform_fn(data_item):
        return data_item.unsqueeze(0)

    calibration_dataset = nncf.Dataset(calibration_images, transform_fn)

    # --- THE ACCURACY SHIELD ---
    # We apply the same pattern-based shield that worked in convert.py
    # to protect the decision-making layers from precision loss.
    ignored_scope = nncf.IgnoredScope(
        patterns=[".*fc.*", ".*output.*"]
    )

    print("\n🚀 Running MIXED-Precision Quantization (NPU Optimized)...")
    start_time = time.time()

    try:
        quantized_model = nncf.quantize(
            model,
            calibration_dataset,
            preset=nncf.QuantizationPreset.MIXED, # Best for OCR/Alphanumeric
            subset_size=1000,
            ignored_scope=ignored_scope,
            fast_bias_correction=True
        )
    except Exception as e:
        print(f"⚠️ Accuracy Shielding error: {e}. Falling back to standard MIXED.")
        quantized_model = nncf.quantize(
            model,
            calibration_dataset,
            preset=nncf.QuantizationPreset.MIXED,
            subset_size=1000,
            fast_bias_correction=True
        )

    print(f"\n✅ Optimization Finished in {time.time() - start_time:.1f}s")

    # --- EXPORT TO OPENVINO ---
    print("💾 Saving Static INT8 model (200x80) for NPU...")
    example_input = torch.randn(1, 1, config.HEIGHT, config.WIDTH)
    ov_model = ov.convert_model(quantized_model, example_input=example_input)

    # Lock the shape for Arrow Lake NPU hardware acceleration
    ov_model.reshape({0: [1, 1, config.HEIGHT, config.WIDTH]})

    xml_path = os.path.join(output_dir, "captcha_model_int8.xml")
    ov.save_model(ov_model, xml_path)

    print("-" * 50)
    print(f"🏁 DONE! NPU-Ready model saved: {xml_path}")
    print("-" * 50)

if __name__ == "__main__":
    run_quantization()
