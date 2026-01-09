"""
test solving fresh captchas using the NPU and model
"""
import os
import sys
import time
import openvino as ov
import numpy as np
from captcha.image import ImageCaptcha

# --- 1. SETUP & CONFIG ---
try:
    from config import CHARS, CAPTCHA_LENGTH, WIDTH, HEIGHT
except ImportError:
    # Advanced 62-char set: 0-9, a-z, A-Z
    CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    CAPTCHA_LENGTH = 6
    WIDTH, HEIGHT = 200, 80

core = ov.Core()
TEST_SAMPLES = 100
generator = ImageCaptcha(width=WIDTH, height=HEIGHT)

# Updated Path: Matches the 'advanced' directory structure
current_dir = os.path.dirname(os.path.abspath(__file__))
model_xml = os.path.join(current_dir, "openvino_int8_model", "captcha_model_int8.xml")

# --- 2. LOAD NPU ENGINE ---
print("🚀 Initializing Intel AI Boost (Arrow Lake NPU)...")
try:
    if not os.path.exists(model_xml):
        raise FileNotFoundError(f"Missing INT8 model at {model_xml}")

    model = core.read_model(model_xml)
    # Force static shape to ensure NPU optimization
    model.reshape({0: [1, 1, HEIGHT, WIDTH]})
    compiled_model = core.compile_model(model, "NPU")
    print("✅ NPU Engine Ready. INT8 Advanced Alphanumeric Model Loaded.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit()

# --- 3. VALIDATION LOOP ---
print(f"📊 Testing {TEST_SAMPLES} live-generated {CAPTCHA_LENGTH}-char advanced CAPTCHAs...")
success_count = 0
total_inference_time = 0

for i in range(TEST_SAMPLES):
    # Generate random secret using the full 62-character set
    secret = "".join([np.random.choice(list(CHARS)) for _ in range(CAPTCHA_LENGTH)])

    # Generate Image (Must be 'L' for Grayscale)
    img = generator.generate_image(secret).convert('L')

    # Preprocess (Matches training transforms exactly)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = (img_np - 0.5) / 0.5  # Standardization
    input_tensor = img_np.reshape(1, 1, HEIGHT, WIDTH)

    # NPU Inference
    start = time.perf_counter()
    results = compiled_model([input_tensor])[0]
    end = time.perf_counter()

    total_inference_time += (end - start)

    # Decode Output
    # Advanced Model Output: [1, 372] -> Reshape to [6, 62]
    predictions = results.reshape(CAPTCHA_LENGTH, -1)
    pred_indices = np.argmax(predictions, axis=1)
    pred_str = "".join([CHARS[idx] for idx in pred_indices])

    if pred_str == secret:
        success_count += 1

    if (i + 1) % 10 == 0:
        print(f"  > Benchmarking: {i + 1}/{TEST_SAMPLES}...")

# --- 4. FINAL NPU REPORT ---
success_rate = (success_count / TEST_SAMPLES) * 100
avg_time_ms = (total_inference_time / TEST_SAMPLES) * 1000

print("\n" + "="*45)
print("           ADVANCED NPU TEST REPORT")
print("="*45)
print("Hardware Engine:     Intel® AI Boost (NPU)")
print("Character Set:       62 (Alphanumeric Mixed-Case)")
print(f"Input Resolution:    {WIDTH}x{HEIGHT}")
print(f"Success Rate:        {success_rate:.2f}%")
print(f"Avg NPU Latency:     {avg_time_ms:.2f} ms")
print(f"Peak Throughput:     {1000/avg_time_ms:.1f} Captchas/Sec")
print("="*45)

if success_rate > 95:
    print("\n🏆 Production Ready: NPU accuracy exceeds high-performance threshold.")
elif success_rate > 80:
    print("\n⚠️  Stable: Good performance, but check for case-sensitivity confusions.")
else:
    print("\n💡 Tip: Accuracy drop detected. Consider increasing INT8 calibration samples.")
