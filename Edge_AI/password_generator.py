import os
import time
import json
import csv
import re
import string
import openvino_genai as ov_genai

# --- CRITICAL HARDWARE FLAGS ---
os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"

def check_password_complexity(pwd):
    """Validation for 8+ chars and 3 of 4 classes."""
    if not pwd or len(pwd) < 8: return False
    categories = [
        any(c.islower() for c in pwd),
        any(c.isupper() for c in pwd),
        any(c.isdigit() for c in pwd),
        any(not c.isalnum() for c in pwd)
    ]
    return sum(categories) >= 3

def format_persona_prompt():
    """Builds the persona-specific query."""
    system_msg = "You are a creative identity generator. You must output only raw JSON."
    user_query = (
        "Generate a unique persona with name, city, birthdate, zodiac, job, "
        "and three passwords (personal_email, work_pc, banking). "
        "The work_pc password MUST have 8+ characters and use 3 of 4 character types "
        "(Uppercase, Lowercase, Numbers, Symbols)."
    )
    return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"

def run_persona_generation(total_count=100):
    model_path = os.path.join(os.environ['USERPROFILE'], 'Edge-AI', 'models', 'qwen-1.5b')
    cache_dir = os.path.join(os.getcwd(), 'npu_cache')
    output_file = "npu_personas.json"

    # Configuration identical to your working test script
    pipeline_config = {
        "MAX_PROMPT_LEN": 1024,
        "MIN_RESPONSE_LEN": 128,
        "PREFILL_HINT": "STATIC",
        "GENERATE_HINT": "BEST_PERF",
        "PERFORMANCE_HINT": "LATENCY",
        "CACHE_DIR": cache_dir
    }

    print(f"Initializing NPU Pipeline for {total_count} personas...")
    try:
        pipe = ov_genai.LLMPipeline(model_path, "NPU", **pipeline_config)
    except Exception as e:
        print(f"Initial NPU Error: {e}")
        return

    results = []

    for i in range(total_count):
        valid_entry = False
        attempts = 0
        
        while not valid_entry and attempts < 5:
            attempts += 1
            full_response = ""
            
            # Use your proven streamer logic
            def streamer(subword):
                nonlocal full_response
                full_response += subword
                return ov_genai.StreamingStatus.RUNNING

            print(f"[{i+1}/{total_count}] Generating (Attempt {attempts})...", end="\r")
            
            try:
                # Direct generate call using your verified method
                pipe.generate(format_persona_prompt(), max_new_tokens=512, streamer=streamer, do_sample=True, temperature=0.8)
                
                # Extract JSON block
                json_match = re.search(r'\{.*\}', full_response.replace('\n', ' '), re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    work_pwd = data.get("work_pc", "")
                    
                    if check_password_complexity(work_pwd):
                        results.append(data)
                        valid_entry = True
                        print(f"[{i+1}/{total_count}] Success: {data.get('name')}")
                        # Incremental save
                        with open(output_file, "w") as f:
                            json.dump(results, f, indent=4)
                    else:
                        continue # Password didn't meet complexity
            except Exception as e:
                print(f"\nStream Error: {e}")
                time.sleep(1)

    print(f"\nFinished! Results saved to {output_file}")

if __name__ == "__main__":
    run_persona_generation(100)
