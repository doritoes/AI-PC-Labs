"""
benchmark solving CAPTCHAs using the quantized model
"""
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
TARGET_SUCCESSES = 100  # We now target 100 SUCCESSES, not 100 attempts
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
attempts = 0
strikes = 0
start_time = time.perf_counter()

header = f"{'HACKING PROGRESS':<20} | {'TARGET':<8} | {'PREDICT':<8} | {'STAT':<6} | {'STRIKES':<8} | {'ACCURACY'}"
print(header)
print("-" * len(header))

# Changed to while loop to target successful cracks
while successes < TARGET_SUCCESSES:
    attempts += 1
    secret = "".join([np.random.choice(list(CHARS)) for _ in range(CAPTCHA_LENGTH)])
    img = generator.generate_image(secret).convert('L')

    # Preprocess
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

    # Formatting: Percentage is based on SUCCESSES toward TARGET
    progress_str = f"{(successes / TARGET_SUCCESSES * 100):.0f}% Cracked..."
    acc_pct = (successes / attempts) * 100
    strike_display = "!" * strikes if strikes > 0 else "-"
    
    print(f"{progress_str:<20} | {secret:<8} | {pred_str:<8} | {status:<6} | {strike_display:<8} | {acc_pct:.1f}%")

    if strikes >= MAX_STRIKES:
        print("\n" + "!" * 60)
        print("!! SECURITY ALERT: PERSISTENT ANOMALY DETECTED !!")
        print("!! FIREWALL ACTIVE - NPU SIGNATURE BLACKLISTED !!")
        print("!" * 60)
        print(f"""
          ________________________________________________
         /                                                \\
        |    [!]  SYSTEM LOCKOUT ENGAGED  [!]              |
        |                                                  |
        |    CONSECUTIVE FAILURES: {strikes}                       |
        |    SESSION ID: NPU-BREACH-ERROR-808              |
        |    STATUS: TERMINATED BY HOST                    |
         \\________________________________________________/
        """)
        print("... Connection Lost ...")
        sys.exit()

duration = time.perf_counter() - start_time

# --- 3. THE VERDICT ---
print("\n" + "═" * len(header))
if duration <= 10.0:
    print(f"🏆  MISSION SUCCESS: DATA BREACH COMPLETE")
    print(f"    Total Attempts: {attempts}")
    print(f"    Final Accuracy: {acc_pct:.2f}%")
    print(f"    Execution Time: {duration:.4f}s")
    print(f"    Throughput:     {attempts/duration:.2f} caps/sec")
    print("═" * len(header))
    print("  >>> ACCESS GRANTED: QUANTUM GATEWAY BYPASSED <<<")
else:
    print(f"⚠️   MISSION FAILED: SYSTEM REJECTED ACCESS")
    print(f"    Reason: Latency Threshold Not Met ({duration:.2f}s > 10s)")
    print("═" * len(header))
print("\n")
