import os
import sys
import time
import openvino as ov
import numpy as np
from captcha.image import ImageCaptcha

# --- 1. CONFIG & NPU SYNC ---
try:
    from config import CHARS, CAPTCHA_LENGTH, WIDTH, HEIGHT
except ImportError:
    CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    CAPTCHA_LENGTH = 6
    WIDTH, HEIGHT = 200, 80

core = ov.Core()
TARGET_ATTEMPTS = 100
MAX_STRIKES = 5 
generator = ImageCaptcha(width=WIDTH, height=HEIGHT)

model_xml = os.path.join(os.path.dirname(__file__), "final_npu_int8.xml")

print("\n" + "⚡" * 30)
print("⚡  NPU SABOTEUR PROTOCOL: ACTIVATED")
print("⚡  HARDWARE: INTEL® AI BOOST (ARROW LAKE)")
print("⚡" * 30)

try:
    model = core.read_model(model_xml)
    model.reshape({0: [1, 1, HEIGHT, WIDTH]})
    compiled_model = core.compile_model(model, "NPU")
    print(f"✅ NEURAL ENGINE SYNCED. READY FOR BREACH.\n")
except Exception as e:
    print(f"❌ LINK FAILURE: {e}")
    sys.exit()

# --- 2. THE BREACH LOOP ---
successes = 0
strikes = 0
start_time = time.perf_counter()

print(f"{'ITER':<6} | {'TARGET':<8} | {'PREDICT':<8} | {'STAT':<5} | {'STRIKES':<8} | {'ACCURACY'}")
print("-" * 75)

for i in range(1, TARGET_ATTEMPTS + 1):
    secret = "".join([np.random.choice(list(CHARS)) for _ in range(CAPTCHA_LENGTH)])
    img = generator.generate_image(secret).convert('L')

    # Preprocess (Standard 0-1 for the 63% model)
    img_np = np.array(img).astype(np.float32) / 255.0
    input_tensor = img_np.reshape(1, 1, HEIGHT, WIDTH)

    # NPU Inference
    results = compiled_model([input_tensor])[0]
    predictions = results.reshape(6, 62)
    pred_str = "".join([CHARS[idx] for idx in np.argmax(predictions, axis=1)])

    # Logic Check
    if pred_str == secret:
        successes += 1
        strikes = 0
        status = "✅ WIN"
    else:
        strikes += 1
        status = "❌ FAIL"

    # Real-time Per-Attempt Logging
    acc_pct = (successes / i) * 100
    strike_display = "!" * strikes if strikes > 0 else "-"
    print(f"{i:<6} | {secret:<8} | {pred_str:<8} | {status:<5} | {strike_display:<8} | {acc_pct:.1f}%")

    if strikes >= MAX_STRIKES:
        print("\n" + "█" * 60)
        print("█  CRITICAL ERROR: FIREWALL DETECTED NPU SIGNATURE  █")
        print(f"█  STRIKE LIMIT EXCEEDED ({MAX_STRIKES}) - LOCKOUT ENGAGED  █")
        print("█" * 60)
        sys.exit()

duration = time.perf_counter() - start_time

# --- 3. THE VERDICT ---
print("\n" + "=" * 60)
if duration <= 10.0 and successes >= 60:
    print(f"🏆 MISSION SUCCESS: DATA BREACH COMPLETE")
    print(f"   Final Accuracy: {successes}%")
    print(f"   Execution Time: {duration:.4f}s")
    print(f"   Throughput:     {TARGET_ATTEMPTS/duration:.2f} caps/sec")
else:
    print(f"⚠️  MISSION FAILED: SYSTEM REJECTED ACCESS")
    print(f"   Reason: {'Latency' if duration > 10 else 'Accuracy'} Threshold Not Met")
print("=" * 60 + "\n")
