import openvino as ov
import numpy as np
import time
import os
from captcha.image import ImageCaptcha

# 1. Initialize
core = ov.Core()
CHARS = "0123456789"

# 2. Load the INT8 Model to the NPU
print("Loading INT8 model to Intel AI Boost NPU...")
model = core.read_model("captcha_model_int8.xml")
compiled_model = core.compile_model(model, "NPU")

# 3. Setup a "Brute Force" batch
# We'll run 10 different CAPTCHAs to see the NPU performance
generator = ImageCaptcha(width=160, height=60)
total_time = 0
attempts = 10

print(f"Starting NPU-accelerated attack ({attempts} samples)...")
print("-" * 40)

for i in range(attempts):
    # Generate random label
    secret = "".join([np.random.choice(list(CHARS)) for _ in range(4)])
    
    # Preprocess
    img = generator.generate_image(secret)
    img_np = np.array(img.convert('L')) / 255.0
    input_tensor = np.expand_dims(np.expand_dims(img_np, 0), 0).astype(np.float32)

    # Infer
    start = time.perf_counter()
    results = compiled_model([input_tensor])[compiled_model.output(0)]
    end = time.perf_counter()
    
    total_time += (end - start)

    # Decode
    pred = results.reshape(4, 10).argmax(axis=1)
    pred_str = "".join([CHARS[idx] for idx in pred])
    
    status = "SUCCESS" if pred_str == secret else "FAIL"
    print(f"[{i+1}] Target: {secret} | Predicted: {pred_str} | {status} | {(end-start)*1000:.2f}ms")

print("-" * 40)
print(f"Average NPU Inference Time: {(total_time/attempts)*1000:.2f} ms")
print(f"Total NPU Throughput: {attempts/total_time:.2f} CAPTCHAs/sec")
