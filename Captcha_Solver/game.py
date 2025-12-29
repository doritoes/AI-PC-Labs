import openvino as ov
import numpy as np
import time
import sys
from captcha.image import ImageCaptcha

# --- 1. INITIALIZATION ---
core = ov.Core()
CHARS = "0123456789"
TARGET_SOLVES = 100
MAX_STRIKES = 5  # 5 mistakes in a row = Game Over
generator = ImageCaptcha(width=160, height=60)

# Load NPU Model
print("Initializing NPU Hacking Module...")
model = core.read_model("captcha_model_int8.xml")
compiled_model = core.compile_model(model, "NPU")

# --- 2. GAME LOOP ---
print("\n" + "!"*40)
print(" SYSTEM BREACH INITIALIZED")
print(f" GOAL: Solve {TARGET_SOLVES} CAPTCHAs in < 5s")
print(f" RULE: {MAX_STRIKES} consecutive errors = LOCKOUT")
print("!"*40 + "\n")

solves = 0
strikes = 0
start_time = time.perf_counter()

try:
    for i in range(1, TARGET_SOLVES + 1):
        # Generate random CAPTCHA
        secret = "".join([np.random.choice(list(CHARS)) for _ in range(4)])
        img = generator.generate_image(secret)
        
        # Preprocess for NPU
        img_np = np.array(img.convert('L')) / 255.0
        input_tensor = np.expand_dims(np.expand_dims(img_np, 0), 0).astype(np.float32)

        # Inference
        results = compiled_model([input_tensor])[compiled_model.output(0)]
        pred = results.reshape(4, 10).argmax(axis=1)
        pred_str = "".join([CHARS[idx] for idx in pred])

        # Check Result
        if pred_str == secret:
            solves += 1
            strikes = 0  # Reset consecutive strike counter on success
            print(f"[SUCCESS] {i:03d}: {pred_str} | Breach Progress: {solves}%", end="\r")
        else:
            strikes += 1
            print(f"\n[ ALERT ] Strike {strikes}/{MAX_STRIKES} - Failed: {secret} != {pred_str}")
            
        if strikes >= MAX_STRIKES:
            print("\n" + "-"*40)
            print("!!! SECURITY LOCKOUT TRIGGERED !!!")
            print("Reason: Too many consecutive failed attempts.")
            print("-" * 40)
            sys.exit()

    end_time = time.perf_counter()
    duration = end_time - start_time

    # --- 3. RESULTS ---
    print("\n\n" + "="*40)
    if duration <= 5.0:
        print(" MISSION ACCOMPLISHED: SYSTEM BYPASSED")
    else:
        print(" MISSION FAILED: TIME LIMIT EXCEEDED")
    
    print("="*40)
    print(f"Total Time:    {duration:.2f} seconds")
    print(f"Total Solves:  {solves}/{TARGET_SOLVES}")
    print(f"Avg Speed:     {duration/TARGET_SOLVES*1000:.2f} ms/captcha")
    print("="*40)

except KeyboardInterrupt:
    print("\nDisconnecting...")
