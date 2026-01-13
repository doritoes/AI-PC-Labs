"""
benchmark solving CAPTCHAs using the quantized model
"""
import os
import time
import numpy as np
import openvino as ov
from captcha.image import ImageCaptcha
from torchvision import transforms
import config

def decode(logits):
    """Convert raw NPU output into characters."""
    logits = logits.reshape(config.CAPTCHA_LENGTH, -1)
    return "".join([config.CHARS[np.argmax(c)] for c in logits])

def run_final_breach(num_tests=100):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "final_npu_int8.xml")

    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} not found! Run quantize-improved.py first.")
        return

    # 1. Initialize NPU
    print("🚀 Initializing Intel® AI Boost (NPU)...")
    core = ov.Core()
    compiled_model = core.compile_model(model_path, "NPU")
    infer_request = compiled_model.create_infer_request()

    # 2. Setup Generator (Matches 0.0-1.0 Range used in Quantization)
    generator = ImageCaptcha(width=200, height=80)
    transform = transforms.Compose([transforms.ToTensor()])

    success_count = 0
    latencies = []

    print(f"📊 Running {num_tests}-Captcha Breach Test...")
    print("-" * 50)

    # 3. Execution Loop
    for i in range(1, num_tests + 1):
        # Generate target
        real_text = ''.join(np.random.choice(list(config.CHARS), config.CAPTCHA_LENGTH))
        img = generator.generate_image(real_text).convert('L')

        # Pre-process
        input_tensor = transform(img).unsqueeze(0).numpy()

        # NPU Inference
        start_time = time.perf_counter()
        results = infer_request.infer({0: input_tensor})
        end_time = time.perf_counter()

        # Post-process
        output_node = compiled_model.outputs[0]
        prediction = decode(results[output_node])

        # Scoring
        duration = (end_time - start_time) * 1000
        latencies.append(duration)

        is_correct = prediction == real_text
        if is_correct:
            success_count += 1

        # Live feedback every 10 iterations
        if i % 10 == 0:
            current_acc = (success_count / i) * 100
            print(f"  > [{i}/{num_tests}] | Accuracy: {current_acc:.1f}% | Latency: {duration:.2f}ms")

    # 4. Final Report
    avg_latency = np.mean(latencies)
    total_time = sum(latencies) / 1000
    final_accuracy = (success_count / num_tests) * 100

    print("\n" + "="*45)
    print("           FINAL NPU MISSION REPORT")
    print("="*45)
    print(f"Total Solved:        {success_count}/{num_tests}")
    print(f"Final Accuracy:      {final_accuracy:.2f}%")
    print(f"Avg NPU Latency:     {avg_latency:.2f} ms")
    print(f"Total Sequence Time: {total_time:.2f} seconds")
    print("="*45)

    if final_accuracy >= 60:
        print("\n🔥 STATUS: MISSION SUCCESS. NPU OPTIMIZED.")
    else:
        print("\n⚠️ STATUS: ACCURACY DROP. CHECKING CHARACTER CONFUSION...")

if __name__ == "__main__":
    run_final_breach()
