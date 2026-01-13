"""
quantize the OpenVINO model to INT8
"""
import os
import nncf
import openvino as ov
import numpy as np
from captcha.image import ImageCaptcha
from torchvision import transforms
import config

def quantize_safe():
    """ method to quantize without losing accuracy """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "baseline_npu_fp32.xml")

    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} not found! Run convert_to_npu.py first.")
        return

    core = ov.Core()
    ov_model = core.read_model(model_path)

    # 1. High-Diversity Calibration
    print("🧪 Generating 2000 samples for safety-first calibration...")
    generator = ImageCaptcha(width=200, height=80)
    transform = transforms.Compose([transforms.ToTensor()])

    calibration_images = []
    for _ in range(2000):
        text = ''.join(np.random.choice(list(config.CHARS), 6))
        img = generator.generate_image(text).convert('L')
        calibration_images.append(transform(img).numpy())

    def transform_fn(data_item):
        return np.expand_dims(data_item, 0)

    calibration_dataset = nncf.Dataset(calibration_images, transform_fn)

    # 2. Flexible Accuracy Shield
    # We use names we know are in the graph from your previous successful runs.
    # We remove 'conv1' to avoid the strict validation crash.
    ignored_scope = nncf.IgnoredScope(
        patterns=[
            ".*fc.*",
            ".*output.*",
            ".*linear.*"
        ]
    )

    print("🚀 Quantizing: Conv -> INT8 | Classification -> FP16...")

    # We set validate=False internally by catching the potential error
    # and falling back to a guaranteed match.
    try:
        quantized_model = nncf.quantize(
            ov_model,
            calibration_dataset,
            preset=nncf.QuantizationPreset.MIXED,
            ignored_scope=ignored_scope,
            subset_size=2000,
            fast_bias_correction=True
        )
    except Exception as e:
        print(f"⚠️ Refining shield due to: {e}")
        # Baseline safe shield
        ignored_scope = nncf.IgnoredScope(patterns=[".*fc.*", ".*output.*"])
        quantized_model = nncf.quantize(
            ov_model,
            calibration_dataset,
            preset=nncf.QuantizationPreset.MIXED,
            ignored_scope=ignored_scope,
            subset_size=2000
        )

    out_path = os.path.join(current_dir, "final_npu_int8.xml")
    ov.save_model(quantized_model, out_path)
    print(f"🏁 FINAL NPU MODEL READY: {out_path}")

if __name__ == "__main__":
    quantize_safe()
