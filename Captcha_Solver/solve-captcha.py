"""
test solving fresh captchas using the NPU and model
"""
import sys
import time
import openvino as ov
import numpy as np
from captcha.image import ImageCaptcha

# --- 1. SETUP ---
core = ov.Core()
CHARS = "0123456789"
TEST_SAMPLES = 100  # Increased for a statistically significant success rate
generator = ImageCaptcha(width=160, height=60)

# --- 2. LOAD NPU MODEL ---
print("Loading INT8 model to Intel AI Boost NPU...")
try:
    model = core.read_model("captcha_model_int8.xml")
    compiled_model = core.compile_model(model, "NPU")
    print("NPU Engine Ready.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit()

# --- 3. VALIDATION LOOP ---
print(f"Running NPU Validation on {TEST_SAMPLES} new CAPTCHAs...")
success_count = 0
total_inference_time = 0

for i in range(TEST_SAMPLES):
    # Generate random secret
    secret = "".join([np.random.choice(list(CHARS)) for _ in range(4)])

    # Generate and Preprocess Image
    img = generator.generate_image(secret)
    img_np = np.array(img.convert('L')) / 255.0
    input_tensor = np.expand_dims(np.expand_dims(img_np, 0), 0).astype(np.float32)

    # NPU Inference
    start = time.perf_counter()
    results = compiled_model([input_tensor])[compiled_model.output(0)]
    end = time.perf_counter()

    total_inference_time += (end - start)

    # Decode Output
    pred = results.reshape(4, 10).argmax(axis=1)
    pred_str = "".join([CHARS[idx] for idx in pred])

    if pred_str == secret:
        success_count += 1

# --- 4. FINAL REPORT ---
success_rate = (success_count / TEST_SAMPLES) * 100
avg_time_ms = (total_inference_time / TEST_SAMPLES) * 1000

print("\n" + "="*40)
print("             NPU TEST REPORT")
print("="*40)
print("Hardware Engine:     Intel AI Boost (NPU)")
print("Model Precision:     INT8 (Quantized)")
print(f"Total Samples:       {TEST_SAMPLES}")
print(f"Successful Cracks:   {success_count}")
print(f"NPU Success Rate:    {success_rate:.2f}%")
print(f"Avg Inference Time:  {avg_time_ms:.2f} ms")
print(f"Throughput:          {TEST_SAMPLES/total_inference_time:.2f} items/sec")
print("="*40)

# Compare back to your training script result
print("\nVerification: If this is within 1-2% of your CPU training")
print("accuracy, the quantization was successful.")
