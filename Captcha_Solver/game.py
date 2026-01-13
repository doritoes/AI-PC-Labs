"""
simulate a hacking game based on solving captchas
"""
import os
import sys
import time
import openvino as ov
import numpy as np
from captcha.image import ImageCaptcha

# --- 1. INITIALIZATION ---
core = ov.Core()
CHARS = "0123456789"
TARGET_ATTEMPTS = 100 # We want to survive 100 attempts
MAX_STRIKES = 5
LEADERBOARD_FILE = "leaderboard.txt"
generator = ImageCaptcha(width=160, height=60)

print("Initializing NPU Saboteur Module...")
model = core.read_model("captcha_model_int8.xml")
compiled_model = core.compile_model(model, "NPU")

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
print("\n" + "!"*40)
print(" SYSTEM BREACH INITIALIZED")
print(f" GOAL: Complete {TARGET_ATTEMPTS} attempts in < 5s")
print(f" RULE: {MAX_STRIKES} consecutive errors = LOCKOUT")
print("!"*40 + "\n")

successes = 0
strikes = 0
start_time = time.perf_counter()

for i in range(1, TARGET_ATTEMPTS + 1):
    secret = "".join([np.random.choice(list(CHARS)) for _ in range(4)])
    img = generator.generate_image(secret)

    img_np = np.array(img.convert('L')) / 255.0
    input_tensor = np.expand_dims(np.expand_dims(img_np, 0), 0).astype(np.float32)

    results = compiled_model([input_tensor])[compiled_model.output(0)]
    pred = results.reshape(4, 10).argmax(axis=1)
    pred_str = "".join([CHARS[idx] for idx in pred])

    if pred_str == secret:
        successes += 1
        strikes = 0
        sys.stdout.write(f"\r[PROGRESS] {i}% complete... (Successes: {successes})")
        sys.stdout.flush()
    else:
        strikes += 1
        print(f"\n[ STRIKE {strikes} ] Bypass Failed: {secret} != {pred_str}")

    if strikes >= MAX_STRIKES:
        print("\n" + "="*40)
        print("!!! LOCKOUT: FIREWALL DETECTED PATTERN !!!")
        print("="*40)
        sys.exit()

duration = time.perf_counter() - start_time

# --- 3. FINAL VERDICT ---
print("\n\n" + "="*40)
if duration <= 5.0:
    print("  ACCESS GRANTED: GATEWAY BYPASSED")
    rank, best = update_leaderboard(duration)
    print(f"  Accuracy: {successes}% | Rank: #{rank}")
else:
    print("  MISSION FAILED: TIMEOUT")

print(f"  Final Time: {duration:.4f}s")
print("="*40)
