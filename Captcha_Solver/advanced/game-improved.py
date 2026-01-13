import os
import sys
import time
import openvino as ov
import numpy as np
from captcha.image import ImageCaptcha

# --- 1. INITIALIZATION & CONFIG ---
try:
    from config import CHARS, CAPTCHA_LENGTH, WIDTH, HEIGHT
except ImportError:
    CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    CAPTCHA_LENGTH = 6
    WIDTH, HEIGHT = 200, 80

core = ov.Core()
TARGET_ATTEMPTS = 100
MAX_STRIKES = 5  # Increased slightly because 63% accuracy means more natural misses
LEADERBOARD_FILE = "leaderboard_advanced.txt"
generator = ImageCaptcha(width=WIDTH, height=HEIGHT)

# Path to your NEWly successful INT8 model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_xml = os.path.join(current_dir, "final_npu_int8.xml")

print("\n" + "⚡" * 25)
print("⚡ INITIALIZING ADVANCED NPU SABOTEUR MODULE 2.0")
print("⚡" * 25)

try:
    if not os.path.exists(model_xml):
        raise FileNotFoundError(f"Optimized INT8 Model not found at {model_xml}")
    
    # Load for Arrow Lake NPU
    model = core.read_model(model_xml)
    model.reshape({0: [1, 1, HEIGHT, WIDTH]})
    compiled_model = core.compile_model(model, "NPU")
    
    device_name = core.get_property('NPU', 'FULL_DEVICE_NAME')
    print(f"✅ NPU SYNCHRONIZED: {device_name}")
except Exception as e:
    print(f"❌ CRITICAL INITIALIZATION FAILURE: {e}")
    sys.exit()

def update_leaderboard(player_time):
    scores = []
    if os.path.exists(LEADERBOARD_FILE):
        with open(LEADERBOARD_FILE, "r") as f:
            for line in f:
                try: scores.append(float(line.strip()))
                except: continue
    scores.append(player_time)
    scores.sort()
    with open(LEADERBOARD_FILE, "w") as f:
        for s in scores[:10]:
            f.write(f"{s:.4f}\n")
    return scores.index(player_time) + 1, scores[0]

# --- 2. THE BREACH ---
print("\n" + "!"*50)
print(" !!! ADVANCED SYSTEM BREACH INITIALIZED !!!")
print(f" COMPLEXITY: 6-Chars | Set: Alphanumeric (62)")
print(f" TARGET: {TARGET_ATTEMPTS} attempts in < 10.0s")
print(f" TOLERANCE: {MAX_STRIKES} consecutive errors")
print("!"*50 + "\n")

successes = 0
strikes = 0
start_time = time.perf_counter()

for i in range(1, TARGET_ATTEMPTS + 1):
    secret = "".join([np.random.choice(list(CHARS)) for _ in range(CAPTCHA_LENGTH)])
    img = generator.generate_image(secret).convert('L')

    # PREPROCESS: Using the 0.0 - 1.0 range that yielded your 63% result
    img_np = np.array(img).astype(np.float32) / 255.0
    input_tensor = img_np.reshape(1, 1, HEIGHT, WIDTH)

    # NPU INFERENCE
    results = compiled_model([input_tensor])[0]
    
    # DECODE
    predictions = results.reshape(6, 62)
    pred_indices = np.argmax(predictions, axis=1)
    pred_str = "".join([CHARS[idx] for idx in pred_indices])

    if pred_str == secret:
        successes += 1
        strikes = 0
        sys.stdout.write(f"\r[STATUS] {i}% Cracked... Success Rate: {successes/i*100:.1f}%")
        sys.stdout.flush()
    else:
        strikes += 1
        # Optional: uncomment the line below to see exactly what missed
        # print(f"\n[ STRIKE {strikes} ] Breach Failed at {i}%: {secret} != {pred_str}")

    if strikes >= MAX_STRIKES:
        print("\n\n" + "═"*50)
        print("!!! SECURITY ALERT: FIREWALL DETECTED MALICIOUS NPU PATTERN !!!")
        print("!!! LOCKOUT ENGAGED - SESSION TERMINATED !!!")
        print("═"*50)
        sys.exit()

duration = time.perf_counter() - start_time

# --- 3. THE VERDICT ---
print("\n\n" + "═"*50)
# Adjusting win condition: 60%+ accuracy is now the target
if duration <= 10.0 and successes >= 60:
    print("  🏆 ACCESS GRANTED: QUANTUM GATEWAY BYPASSED")
    rank, best = update_leaderboard(duration)
    print(f"  Accuracy: {successes}% | Rank: #{rank} | Best: {best:.4f}s")
else:
    if duration > 10.0:
        print(f"  ⚠️ MISSION FAILED: THROUGHPUT CRITICAL ({duration:.2f}s > 10s)")
    else:
        print(f"  ⚠️ MISSION FAILED: ACCURACY BELOW THRESHOLD ({successes}%)")

print(f"  Final Time: {duration:.4f}s")
print("═"*50)
