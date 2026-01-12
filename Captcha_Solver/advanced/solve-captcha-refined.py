import os
import time
import numpy as np
import openvino as ov
from captcha.image import ImageCaptcha
from torchvision import transforms
import config

def decode(logits):
    # Reshape to 6 chars x 62 classes
    logits = logits.reshape(config.CAPTCHA_LENGTH, -1)
    return "".join([config.CHARS[np.argmax(c)] for c in logits])

def run_test():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "openvino_models", "refined_model_int8.xml")
    
    core = ov.Core()
    compiled_model = core.compile_model(model_path, "NPU")
    infer_request = compiled_model.create_infer_request()
    
    # MUST MATCH QUANTIZATION: ToTensor() only, no manual Normalize
    transform = transforms.Compose([transforms.ToTensor()])
    generator = ImageCaptcha(width=config.WIDTH, height=config.HEIGHT)

    success = 0
    total = 100
    latencies = []

    print(f"🚀 NPU Test Started (Range: 0.0-1.0)...")

    for _ in range(total):
        text = ''.join(np.random.choice(list(config.CHARS), config.CAPTCHA_LENGTH))
        img = generator.generate_image(text).convert('L')
        input_data = transform(img).unsqueeze(0).numpy()

        start = time.perf_counter()
        res = infer_request.infer({0: input_data})
        latencies.append((time.perf_counter() - start) * 1000)

        pred = decode(list(res.values())[0])
        if pred == text: success += 1

    print(f"\n" + "="*30)
    print(f"RESULT: {success}/{total} ({success}%)")
    print(f"LATENCY: {np.mean(latencies):.2f}ms")
    print("="*30)

if __name__ == "__main__":
    run_test()
