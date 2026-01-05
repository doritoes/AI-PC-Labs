import os
import json
import re
import random
import time
from datetime import datetime, timedelta
import openvino_genai as ov_genai

# --- HARDWARE FLAGS ---
os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"
MODEL_PATH = os.path.join(os.environ['USERPROFILE'], 'Edge-AI', 'models', 'qwen-1.5b')

def get_pipe():
    config = {
        "MAX_PROMPT_LEN": 1024,
        "PREFILL_HINT": "STATIC",
        "GENERATE_HINT": "BEST_PERF",
        "PERFORMANCE_HINT": "LATENCY",
        "CACHE_DIR": "npu_cache"
    }
    return ov_genai.LLMPipeline(MODEL_PATH, "NPU", **config)

def get_seeds(pipe, category, count=20):
    """Asks Qwen to generate its own diverse seeds to avoid Gemini bias."""
    prompt = f"<|im_start|>user\nList {count} diverse and unique {category}. Separate by commas. No numbers.<|im_end|>\n<|im_start|>assistant\n"
    response = pipe.generate(prompt, max_new_tokens=512)
    # Clean up the string and split
    items = [i.strip() for i in response.split(",") if len(i.strip()) > 2]
    return items

def run_autonomous_study(total_count=100):
    pipe = get_pipe()
    
    print("--- PHASE 1: BOOTSTRAPPING SEEDS FROM QWEN ---")
    jobs = get_seeds(pipe, "occupations like 'Welder' or 'Surgeon'", 30)
    names = get_seeds(pipe, "full names from different cultures", 30)
    hobbies = get_seeds(pipe, "hobbies like 'Woodworking' or 'Yoga'", 30)
    
    print(f"Seeds generated: {len(jobs)} Jobs, {len(names)} Names.")
    
    results = []
    attempts = 0
    start_time = time.time()

    print(f"--- PHASE 2: ISOLATED PASSWORD EXTRACTION ({total_count} SAMPLES) ---")

    while len(results) < total_count:
        attempts += 1
        name = random.choice(names) if names else "Alex Reed"
        job = random.choice(jobs) if jobs else "Technician"
        hobby = random.choice(hobbies) if hobbies else "Gaming"
        
        # We RE-PROMPT from scratch every time to prevent conversation tangling
        prompt = (
            f"<|im_start|>system\nYou are a security researcher studying human behavior.<|im_end|>\n"
            f"<|im_start|>user\nCreate a profile for {name}, who works as a {job} and loves {hobby}.\n"
            f"Requirements:\n"
            f"1. Name their favorite professional tool.\n"
            f"2. Create a lazy email password based on their hobby.\n"
            f"3. Create a work password (leetspeak version of their tool).\n"
            f"Return as JSON: {{\"name\": \"\", \"tool\": \"\", \"email_pwd\": \"\", \"work_pwd\": \"\"}}<|im_end|>\n"
            f"<|im_start|>assistant\n{{"
        )

        full_response = ""
        def streamer(subword):
            nonlocal full_response
            full_response += subword
            return ov_genai.StreamingStatus.RUNNING

        try:
            print(f"[{len(results)+1}/{total_count}] Testing: {name} | {job}...".ljust(60), end="\r")
            pipe.generate(prompt, max_new_tokens=256, streamer=streamer, do_sample=True, temperature=0.9)
            
            # Clean and Parse
            raw = "{" + full_response.strip()
            if not raw.endswith("}"): raw += "}"
            
            match = re.search(r'(\{.*\})', raw.replace('\n', ' '), re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                if len(data.get("work_pwd", "")) > 4:
                    data["logic_source"] = "stateless_qwen_inference"
                    results.append(data)
                    
                    print(f"[{len(results)}/{total_count}] ✅ {name[:12].ljust(12)} | Tool: {data.get('tool','?').ljust(12)} | Work: {data['work_pwd']}")
                    
                    if len(results) % 5 == 0:
                        with open("qwen_autonomous_study.json", "w") as f:
                            json.dump(results, f, indent=4)
        except:
            continue

    print(f"\nStudy Complete in {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    run_autonomous_study(100)
