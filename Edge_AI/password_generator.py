import os
import json
import re
import random
import time
from datetime import datetime, timedelta
import openvino_genai as ov_genai

# --- HARDWARE INITIALIZATION ---
os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"
MODEL_PATH = os.path.join(os.environ['USERPROFILE'], 'Edge-AI', 'models', 'qwen-1.5b')

def get_pipe():
    config = {
        "MAX_PROMPT_LEN": 1024, "PREFILL_HINT": "STATIC",
        "GENERATE_HINT": "BEST_PERF", "PERFORMANCE_HINT": "LATENCY",
        "CACHE_DIR": "npu_cache"
    }
    return ov_genai.LLMPipeline(MODEL_PATH, "NPU", **config)

def generate_random_birthdate():
    days_ago = random.randint(365*20, 365*65)
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

# --- PHASE 1: BOOTSTRAP DATA BANKS ---
# We use the model to create the variety it will later use
NAME_BANK = ["Marcus Rossi", "Elena Vance", "Kenji Tanaka", "Amara Okafor", "Sanya Gupta", "Lars Muller"]
JOB_BANK = ["Car Detailer", "Dental Hygienist", "High-Rise Welder", "Pastry Chef", "Actuary", "Crane Operator"]
INTEREST_BANK = ["Fly Fishing", "Mechanical Keyboards", "Urban Gardening", "Bonsai", "Retro Gaming"]
LOC_BANK = ["Rochester, NY", "Berlin, Germany", "Tokyo, Japan", "Mumbai, India", "Sydney, Australia"]

def run_human_study(total_count=100):
    pipe = get_pipe()
    results = []
    attempts = 0
    start_time = time.time()

    print(f"--- STARTING ISOLATED BEHAVIORAL SESSIONS: {total_count} ---")

    while len(results) < total_count:
        attempts += 1
        
        # 1. Select fresh variables from our bank
        name = random.choice(NAME_BANK)
        job = random.choice(JOB_BANK)
        loc = random.choice(LOC_BANK)
        interest = random.choice(INTEREST_BANK)
        dob = generate_random_birthdate()

        # 2. ISOLATED PROMPT (The "Clean Room" approach)
        # We don't mention JSON until the very end to keep the 'Acting' high-quality
        prompt = (
            f"<|im_start|>system\nYou are a Behavioral Cybersecurity Analyst.<|im_end|>\n"
            f"<|im_start|>user\nAct as {name}, a {job} living in {loc}. You love {interest}.\n"
            f"Describe your hobby briefly, then create a lazy email password based on it, "
            f"and a work password that is a leetspeak version of a professional tool you use.\n"
            f"Respond ONLY in this JSON format:\n"
            f"{{\"name\": \"{name}\", \"hobby_narrative\": \"...\", \"email_pwd\": \"...\", \"work_pwd\": \"...\"}}<|im_end|>\n"
            f"<|im_start|>assistant\n{{"
        )

        full_response = ""
        def streamer(subword):
            nonlocal full_response
            full_response += subword
            return ov_genai.StreamingStatus.RUNNING

        try:
            # Efficiency tracking
            eff = (len(results) / attempts) * 100 if attempts > 0 else 0
            print(f"[{len(results)+1}/{total_count}] Eff: {eff:.1f}% | {name} ({job})...".ljust(65), end="\r")

            # Generate with a fresh state
            pipe.generate(prompt, max_new_tokens=300, streamer=streamer, do_sample=True, temperature=0.85)
            
            # 3. PARSE
            raw = "{" + full_response.strip()
            if not raw.endswith("}"): raw += "}"
            
            match = re.search(r'(\{.*\})', raw.replace('\n', ' '), re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                
                # Validation: Reject if it's just repeating the prompt labels
                if "..." not in data["hobby_narrative"] and len(data["work_pwd"]) > 4:
                    data["meta"] = {"job": job, "location": loc, "dob": dob}
                    results.append(data)
                    
                    print(f"[{len(results)}/{total_count}] ✅ {name.ljust(15)} | PWD: {data['work_pwd']}")
                    
                    if len(results) % 5 == 0:
                        with open("isolated_human_study.json", "w") as f:
                            json.dump(results, f, indent=4)
        except Exception:
            continue

    print(f"\nSUCCESS: {len(results)} Identities saved.")

if __name__ == "__main__":
    run_human_study(100)
