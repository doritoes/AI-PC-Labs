import sys
import time
import openvino as ov
import numpy as np
import os
from captcha.image import ImageCaptcha

# --- 1. SETUP & CONFIG ---
# Import settings from your central config
try:
    from config import CHARS, CAPTCHA_LENGTH, WIDTH, HEIGHT
except ImportError:
    CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    CAPTCHA_LENGTH = 6
    WIDTH, HEIGHT = 200, 80

core = ov.Core()
TEST_SAMPLES = 100 
generator = ImageCaptcha(width=WIDTH, height=HEIGHT)

# Path to your optimized INT8 model
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_xml = os.path.join(root_dir, "openvino_int8_model", "captcha_model_int8.xml")

# --- 2. LOAD NPU MODEL ---
print(f"🚀 Initializing Intel AI Boost NPU...")
try:
    if not os.path.exists(model_xml):
        raise FileNotFoundError(f"Missing INT8 model at {model_xml}")
        
    model = core.read_model(model_xml)
    compiled_model = core.compile_model(model, "NPU")
    print("✅ NPU Engine Ready. Advanced Alphanumeric Model Loaded.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit()

# --- 3. VALIDATION LOOP ---
print(f"📊 Running NPU Validation on {TEST_SAMPLES} live-generated advanced CAPTCHAs...")
success_count = 0
total_inference_time = 0

for i in range(TEST_SAMPLES):
    # Generate random secret using full CHARS set
    secret = "".join([np.random.choice(list(CHARS)) for _ in range(CAPTCHA_LENGTH)])

    # Generate Image
    img = generator.generate_image(secret)
    
    # Preprocess (Matches train.py logic)
    # 1. Grayscale & Normalize to [0, 1]
    img_np = np.array(img.convert('L')).astype(np.float32) / 255.0
    # 2. Standardize (Normalize -0.5 / 0.5)
    img_np = (img_np - 0.5) / 0.5
    # 3. Add Batch and Channel dims: [1, 1, H, W]
    input_tensor = np.expand_dims(np.expand_dims(img_np, 0), 0)

    # NPU Inference
    start = time.perf_counter()
    results = compiled_model([input_tensor])[compiled_model.output(0)]
    end = time.perf_counter()

    total_inference_time += (end - start)

    # Decode Output: [1, 6, 62] -> Argmax over the 62 characters
    # Results[0] gives us the (6, 62) matrix
    pred_indices = np.argmax(results[0], axis=1)
    pred_str = "".join([CHARS[idx] for idx in pred_indices])

    if pred_str == secret:
        success_count += 1
        
    if (i + 1) % 10 == 0:
        print(f"  > Processed {i + 1}/{TEST_SAMPLES} samples...")

# --- 4. FINAL REPORT ---
success_rate = (success_count / TEST_SAMPLES) * 100
avg_time_ms = (total_inference_time / TEST_SAMPLES) * 1000

print("\n" + "="*45)
print("           ADVANCED NPU TEST REPORT")
print("="*45)
print(f"Hardware Engine:     Intel Core Ultra (NPU)")
print(f"Model Complexity:    {CAPTCHA_LENGTH} Chars (Alphanumeric)")
print(f"Model Precision:     INT8 (Quantized)")
print(f"Total Samples:       {TEST_SAMPLES}")
print(f"Successful Cracks:   {success_count}")
print(f"NPU Success Rate:    {success_rate:.2f}%")
print(f"Avg Inference Time:  {avg_time_ms:.2f} ms")
print(f"Total Throughput:    {TEST_SAMPLES/total_inference_time:.2f} items/sec")
print("="*45)

if success_rate > 90:
    print("\n🏆 High Performance Detected: Model is production ready.")
else:
    print("\n💡 Tip: If success rate is low, increase DATASET_SIZE in config.py and retrain.")
