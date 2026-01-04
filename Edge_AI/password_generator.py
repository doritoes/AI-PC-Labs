import os
import json
import csv
import re
import string
import openvino_genai as ov_genai

# 1. Driver/Memory Stability Flags
os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"
# Force the NPU to use a more stable memory allocation path for MatMul
os.environ["OV_NPU_USE_SRAM"] = "0" 

def check_password_complexity(pwd):
    if len(pwd) < 8: return False
    categories = [
        any(c.islower() for c in pwd),
        any(c.isupper() for c in pwd),
        any(c.isdigit() for c in pwd),
        any(not c.isalnum() for c in pwd)
    ]
    return sum(categories) >= 3

def run_validated_batch(count=100):
    model_path = os.path.join(os.environ['USERPROFILE'], 'Edge-AI', 'models', 'qwen-1.5b')
    
    # 2. Optimized Config for NPU stability on AI-PC
    # We set PREFILL_HINT to STATIC and reduce prompt length to avoid MatMul map errors
    pipeline_config = {
        "MAX_PROMPT_LEN": 512,      # Reduced from 1024 for stability
        "MIN_RESPONSE_LEN": 1,
        "PREFILL_HINT": "STATIC",
        "PERFORMANCE_HINT": "LATENCY"
    }

    try:
        # If NPU continues to fail, change "NPU" to "AUTO" or "CPU" to verify model integrity
        pipe = ov_genai.LLMPipeline(model_path, "NPU", **pipeline_config)
    except RuntimeError as e:
        print(f"NPU Error: {e}")
        print("Switching to CPU for this run...")
        pipe = ov_genai.LLMPipeline(model_path, "CPU")

    results = []

    for i in range(count):
        valid_entry = False
        attempts = 0

        while not valid_entry and attempts < 5:
            # Shortened prompt to reduce memory footprint on NPU
            prompt = "<|im_start|>user\nGenerate persona: {name, age, city, job, fav_animal, password_work(8+ chars, 3 of 4 types)}. Return JSON.<|im_end|>\n<|im_start|>assistant\n"
            
            try:
                response = pipe.generate(prompt, max_new_tokens=256)
                # Cleaning response for JSON parser
                clean_json = re.search(r'\{.*\}', response.strip(), re.DOTALL)
                if not clean_json:
                    raise ValueError("No JSON found")
                
                data = json.loads(clean_json.group(0))
                work_pwd = data.get("password_work", "")

                if check_password_complexity(work_pwd):
                    results.append(data)
                    valid_entry = True
                    print(f"[{i+1}/{count}] Success: {data['name']}")
                else:
                    attempts += 1
            except Exception as e:
                attempts += 1
                print(f"Attempt {attempts} failed: {e}")

    with open("validated_identities.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_validated_batch(5)
