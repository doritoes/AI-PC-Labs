import openvino as ov
import numpy as np
import time
import sys
import os
from captcha.image import ImageCaptcha

# --- 1. INITIALIZATION ---
core = ov.Core()
CHARS = "0123456789"
TARGET_SOLVES = 100
MAX_STRIKES = 5  
LEADERBOARD_FILE = "leaderboard.txt"
generator = ImageCaptcha(width=160, height=60)

# Load NPU Model
print("Initializing NPU Saboteur Module...")
model = core.read_model("captcha_model_int8.xml")
compiled_model = core.compile_model(model, "NPU")

def update_leaderboard(player_time):
    scores = []
    if os.path.exists(LEADERBOARD_FILE):
        with open(LEADERBOARD_FILE, "r") as f:
            for line in f:
                try:
                    scores.append(float(line.strip()))
                except ValueError: continue
    
    scores.append(player_time)
    scores.sort()  # Fastest times first
    
    with open(LEADERBOARD_FILE, "w") as f:
        for s in scores[:10]: # Keep top 10
            f.write(f"{s:.4f}\n")
    
    rank = scores.index(player_time) + 1
    return rank, scores[0]

# --- 2. GAME LOOP ---
print("\n" + "!"*40)
print(" SYSTEM BREACH INITIALIZED")
print(f" GOAL: {TARGET_SOLVES} Solves | TIME: < 5s")
print("!"*40 + "\n")

solves = 0
strikes = 0
start_time = time.perf_counter()

try:
    for i in range(1, TARGET_SOLVES + 1):
        secret = "".join([np.random.choice(list(CHARS)) for _ in range(4)])
        img = generator.generate_image(secret)
        
        img_np = np.array(img.convert('L')) / 255.0
        input_tensor = np.expand_dims(np.expand_dims(img_np, 0), 0).astype(np.float32)

        results = compiled_model([input_tensor])[compiled_model.output(0)]
        pred = results.reshape(4, 10).argmax(axis=1)
        pred_str = "".join([CHARS[idx] for idx in pred])

        if pred_str == secret:
            solves += 1
            strikes = 0
            # Rapid-fire progress update
            sys.stdout.write(f"\r[BREACHING] {solves}% complete...")
            sys.stdout.flush()
        else:
            strikes += 1
            print(f"\n[ STRIKE {strikes} ] Failed: {secret}")
            
        if strikes >= MAX_STRIKES:
            print("\n" + "="*40)
            print("LOCKOUT: The AI was detected by the firewall.")
            print("="*40)
            sys.exit()

    end_time = time.perf_counter()
    duration = end_time - start_time

    # --- 3. FINAL VERDICT ---
    print("\n\n" + "="*40)
    if duration <= 5.0 and solves == TARGET_SOLVES:
        print("  ACCESS GRANTED: GATEWAY BYPASSED")
        rank, best = update_leaderboard(duration)
        print(f"  Your Rank: #{rank} | Record: {best:.4f}s")
    else:
        print("  MISSION FAILED: TOO SLOW")
    
    print("="*40)
    print(f"Final Time: {duration:.4f} seconds")
    print(f"Throughput: {TARGET_SOLVES/duration:.2f} caps/sec")
    print("="*40)

except KeyboardInterrupt:
    print("\nAborting mission...")
