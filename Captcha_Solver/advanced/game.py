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
MAX_STRIKES = 5  
LEADERBOARD_FILE = "leaderboard_advanced.txt"
generator = ImageCaptcha(width=WIDTH, height=HEIGHT)

# Path to the INT8 model optimized for NPU in the 'advanced' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_xml = os.path.join(current_dir, "openvino_int8_model", "captcha_model_int8.xml")

print("⚡ Initializing Advanced NPU Saboteur Module...")
try:
    if not os.path.exists(model_xml):
        raise FileNotFoundError(f"INT8 Model not found at {model_xml}")
    
    # Load and compile specifically for the Arrow Lake NPU
    model = core.read_model(model_xml)
    # Lock the shape for static NPU performance
    model.reshape({0: [1, 1, HEIGHT, WIDTH]})
    compiled_model = core.compile_model(model, "NPU")
    print(f"✅ NPU Engine Synchronized. Device: {core.get_property('NPU', 'FULL_DEVICE_NAME')}")
except Exception as e:
    print(f"❌ Initialization Error: {e}")
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

# --- 2. GAME LOOP ---
print("\n" + "!"*50)
print(" !!! ADVANCED SYSTEM BREACH INITIALIZED !!!")
print(f" COMPLEXITY: 6-Chars | Set: Alphanumeric (62)")
print(f" GOAL: Complete {TARGET_ATTEMPTS} attempts in < 10s")
print(f" RULE: {MAX_STRIKES} consecutive errors = LOCKOUT")
print("!"*50 + "\n")

successes = 0
strikes = 0
start_time = time.perf_counter()

for i in range(1, TARGET_ATTEMPTS + 1):
    # Generate random alphanumeric secret
    secret = "".join([np.random.choice(list(CHARS)) for _ in range(CAPTCHA_LENGTH)])
    
    # Generate as Grayscale to match training
    img = generator.generate_image(secret).convert('L')

    # Preprocess: Must match the labs 'refinement' logic exactly
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = (img_np - 0.5) / 0.5
    
    # Shape for NPU: [Batch, Channel, Height, Width]
    input_tensor = img_np.reshape(1, 1, HEIGHT, WIDTH)

    # Execute NPU Inference
    results = compiled_model([input_tensor])[0]
    
    # DECODE FIX: Reshape the flattened [372] output to [6, 62]
    predictions = results.reshape(CAPTCHA_LENGTH, -1)
    pred_indices = np.argmax(predictions, axis=1)
    pred_str = "".join([CHARS[idx] for idx in pred_indices])

    if pred_str == secret:
        successes += 1
        strikes = 0
        sys.stdout.write(f"\r[PROGRESS] {i}% Cracked... (Current Success Rate: {successes/i*100:.1f}%)")
        sys.stdout.flush()
    else:
        strikes += 1
        print(f"\n[ STRIKE {strikes} ] Breach Failed at {i}%: Expected {secret} but got {pred_str}")

    if strikes >= MAX_STRIKES:
        print("\n" + "="*50)
        print("!!! SECURITY ALERT: FIREWALL DETECTED MALICIOUS NPU PATTERN !!!")
        print("!!! LOCKOUT ENGAGED - SESSION TERMINATED !!!")
        print("="*50)
        sys.exit()

duration = time.perf_counter() - start_time

# --- 3. FINAL VERDICT ---
print("\n\n" + "="*50)
# Win Condition: Under 10 seconds AND 85%+ accuracy
if duration <= 10.0 and successes >= (TARGET_ATTEMPTS * 0.85):
    print("  🏆 ACCESS GRANTED: QUANTUM GATEWAY BYPASSED")
    rank, best = update_leaderboard(duration)
    print(f"  Accuracy: {successes}% | Rank: #{rank} | Best: {best:.4f}s")
else:
    if duration > 10.0:
        print(f"  ⚠️ MISSION FAILED: THROUGHPUT TOO LOW ({duration:.2f}s > 10s)")
    else:
        print(f"  ⚠️ MISSION FAILED: ACCURACY TOO LOW ({successes}%)")

print(f"  Final Time: {duration:.4f}s")
print("="*50)
