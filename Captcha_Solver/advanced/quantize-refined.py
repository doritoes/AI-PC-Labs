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

    # 1. Standardized Calibration
    print("🧪 Generating 1000 samples for High-Fidelity NPU calibration...")
    generator = ImageCaptcha(width=config.WIDTH, height=config.HEIGHT)
    transform = transforms.Compose([transforms.ToTensor()])

    calibration_images = []
    for _ in range(1000):
        text = ''.join(np.random.choice(list(config.CHARS), config.CAPTCHA_LENGTH))
        img = generator.generate_image(text).convert('L')
        calibration_images.append(transform(img).numpy())

    calibration_dataset = nncf.Dataset(calibration_images, lambda x: np.expand_dims(x, 0))

    # 2. THE EXPANDED SHIELD
    # We are now shielding:
    # - The FIRST layer (so we don't lose initial image detail)
    # - The LAST layers (to keep your refined logic sharp)
    ignored_scope = nncf.IgnoredScope(
        patterns=[
            ".*conv1.*",     # Protect the "Eyes"
            ".*fc.*",        # Protect the "Brain"
            ".*output.*"     # Protect the "Voice"
        ]
    )

    print("🚀 Quantizing: Middle Layers -> INT8 | IO Layers -> FP16...")
    quantized_model = nncf.quantize(
        ov_model,
        calibration_dataset,
        preset=nncf.QuantizationPreset.MIXED,
        ignored_scope=ignored_scope,
        subset_size=1000,
        fast_bias_correction=True
    )

    out_path = os.path.join(current_dir, "openvino_models", "refined_model_int8.xml")
    ov.save_model(quantized_model, out_path)
    print(f"🏁 High-Fidelity INT8 Model Ready: {out_path}")

if __name__ == "__main__":
    run_quantization()
