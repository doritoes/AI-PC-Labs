import os
import time
import torch
import numpy as np
import openvino as ov
from captcha.image import ImageCaptcha
from torchvision import transforms
import config

def decode(logit_array):
    """Convert model output logits to text string."""
    chars = ""
    # Reshape to (6 characters, 62 possible characters)
    logit_array = logit_array.reshape(config.CAPTCHA_LENGTH, -1)
    for i in range(config.CAPTCHA_LENGTH):
        idx = np.argmax(logit_array[i])
        chars += config.CHARS[idx]
    return chars

def run_refined_npu_test(num_tests=100):
    # 1. Path Setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "openvino_int8_model", "refined_model_int8.xml")
    
    print("🚀 Initializing Intel AI Boost (NPU) with REFINED Model...")
    
    # 2. Initialize OpenVINO and NPU
    core = ov.Core()
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return

    model_ov = core.read_model(model_path)
    # Compile specifically for NPU
    compiled_model = core.compile_model(model_ov, "NPU")
    infer_request = compiled_model.create_infer_request()

    # 3. Setup Generator and Transforms
    generator = ImageCaptcha(width=config.WIDTH, height=config.HEIGHT)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print(f"✅ NPU Engine Ready. Device: {core.get_property('NPU', 'FULL_DEVICE_NAME')}")
    print(f"📊 Testing {num_tests} live-generated 6-char advanced CAPTCHAs...")

    success_count = 0
    latencies = []

    # 4. Benchmark Loop
    for i in range(1, num_tests + 1):
        # Generate random CAPTCHA
        real_text = ''.join(np.random.choice(list(config.CHARS), config.CAPTCHA_LENGTH))
        img = generator.generate_image(real_text).convert('L')
        
        # Pre-process
        input_tensor = transform(img).unsqueeze(0).numpy()

        # Inference
        start_time = time.perf_counter()
        results = infer_request.infer({0: input_tensor})
        end_time = time.perf_counter()
        
        # Post-process
        output_node = compiled_model.outputs[0]
        prediction = decode(results[output_node])
        
        # Stats
        latencies.append((end_time - start_time) * 1000) # ms
        if prediction == real_text:
            success_count += 1
        
        if i % 10 == 0:
            print(f"  > Benchmarking: {i}/{num_tests}...")

    # 5. Final Report
    avg_latency = sum(latencies) / num_tests
    success_rate = (success_count / num_tests) * 100
    throughput = 1000 / avg_latency

    print("\n" + "="*45)
    print("        REFINED NPU TEST REPORT")
    print("="*45)
    print(f"Hardware Engine:     Intel® AI Boost (NPU)")
    print(f"Model Version:       Refined Alphanumeric INT8")
    print(f"Success Rate:        {success_rate:.2f}%")
    print(f"Avg NPU Latency:     {avg_latency:.2f} ms")
    print(f"Peak Throughput:     {throughput:.2f} Captchas/Sec")
    print("="*45)

    if success_rate < 60:
        print("\n💡 Tip: Accuracy is still low. Try increasing refine.py epochs.")
    else:
        print("\n🔥 STATUS: NPU optimization successful. Ready for Breach.")

if __name__ == "__main__":
    run_refined_npu_test()
