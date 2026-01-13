import nncf
import openvino as ov
import numpy as np
from captcha.image import ImageCaptcha
from torchvision import transforms
import config

def quantize_safe():
    core = ov.Core()
    model = core.read_model("baseline_npu_fp32.xml")

    # 1. High-Diversity Calibration
    print("🧪 Generating 2000 samples for safety-first calibration...")
    generator = ImageCaptcha(width=200, height=80)
    transform = transforms.Compose([transforms.ToTensor()])

    calibration_images = []
    for _ in range(2000):
        text = ''.join(np.random.choice(list(config.CHARS), 6))
        img = generator.generate_image(text).convert('L')
        calibration_images.append(transform(img).numpy())

    dataset = nncf.Dataset(calibration_images, lambda x: np.expand_dims(x, 0))

    # 2. Maximum Protection Shield
    # We ignore the first conv and ALL linear layers to keep the 60% accuracy
    ignored_scope = nncf.IgnoredScope(
        patterns=[".*fc.*", ".*output.*", ".*conv1.*"]
    )

    print("🚀 Quantizing: Conv -> INT8 | IO Layers -> FP16...")
    quantized_model = nncf.quantize(
        model,
        dataset,
        preset=nncf.QuantizationPreset.MIXED,
        ignored_scope=ignored_scope,
        fast_bias_correction=True
    )

    ov.save_model(quantized_model, "final_npu_int8.xml")
    print("🏁 FINAL NPU MODEL READY: final_npu_int8.xml")

if __name__ == "__main__":
    quantize_safe()
