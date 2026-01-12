import os
import nncf
import openvino as ov
import numpy as np
from captcha.image import ImageCaptcha
from torchvision import transforms
import config

def run_quantization():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "openvino_models", "refined_model_fp32.xml")
    
    core = ov.Core()
    ov_model = core.read_model(model_path)

    # 1. Standardized Calibration (0.0 - 1.0 Range Only)
    print("🧪 Generating 1000 samples for NPU calibration...")
    generator = ImageCaptcha(width=config.WIDTH, height=config.HEIGHT)
    transform = transforms.Compose([transforms.ToTensor()])

    def transform_fn(data_item):
        return np.expand_dims(data_item, 0)

    calibration_images = []
    for _ in range(1000):
        text = ''.join(np.random.choice(list(config.CHARS), config.CAPTCHA_LENGTH))
        img = generator.generate_image(text).convert('L')
        calibration_images.append(transform(img).numpy())

    calibration_dataset = nncf.Dataset(calibration_images, transform_fn)

    # 2. Mixed Precision Shield (Crucial for 65% solve rate)
    ignored_scope = nncf.IgnoredScope(patterns=[".*fc.*", ".*output.*"])

    print("🚀 Quantizing: Conv -> INT8 | Classification -> FP16...")
    quantized_model = nncf.quantize(
        ov_model,
        calibration_dataset,
        preset=nncf.QuantizationPreset.MIXED,
        ignored_scope=ignored_scope,
        subset_size=1000
    )

    out_path = os.path.join(current_dir, "openvino_models", "refined_model_int8.xml")
    ov.save_model(quantized_model, out_path)
    print(f"🏁 Refined INT8 Model Ready: {out_path}")

if __name__ == "__main__":
    run_quantization()
