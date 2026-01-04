import os
import json
import re
import string
import time
import openvino_genai as ov_genai

# --- 1. HARDWARE CONFIG ---
os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"

def repair_json_light(raw_text):
    """Fixes common LLM JSON errors like trailing commas or missing braces."""
    try:
        # Extract the core JSON block
        match = re.search(r'(\{.*\})', raw_text.replace('\n', ' '), re.DOTALL)
        if not match:
            return None
        json_str = match.group(1).strip()
        # Fix trailing commas: {"a": 1, } -> {"a": 1}
        json_str = re.sub(r',\s*\}', '}', json_str)
        return json.loads(json_str)
    except:
        return None

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

def run_npu_persona_batch(total_count=100):
    model_path = os.path.join(os.environ['USERPROFILE'], 'Edge-AI', 'models', 'qwen-1.5b')
    cache_dir = os.path.join(os.getcwd(), 'npu_cache')
    output_file = "npu_personas.json"
    
    pipeline_config = {
        "MAX_PROMPT_LEN": 1024,
        "MIN_RESPONSE_LEN": 128,
        "PREFILL_HINT": "STATIC",
        "GENERATE_HINT": "BEST_PERF",
        "PERFORMANCE_HINT": "LATENCY",
        "CACHE_DIR": cache_dir
    }

    print(f"--- INITIALIZING NPU ({total_count} Personas) ---")
    try:
        pipe = ov_genai.LLMPipeline(model_path, "NPU", **pipeline_config)
    except Exception as e:
        print(f"FAILED TO LOAD NPU: {e}")
        return

    results = []
    
    while len(results) < total_count:
        full_response = "{" 
        
        def streamer(subword):
            nonlocal full_response
            full_response += subword
            return ov_genai.StreamingStatus.RUNNING

        # Included all your requested fields in the prompt
        prompt = (
            "<|im_start|>system\nYou are a JSON generator. No chat.<|im_end|>\n"
            "<|im_start|>user\nGenerate a persona with: name, city, birthdate, zodiac, address, "
            "transportation, blood_type, hair_color, eye_color, height, weight, religion, "
            "fav_vacation_spot, fav_season, fav_animal, best_friend, birthplace, "
            "personal_email_pwd, work_pc_pwd, banking_pwd. "
            "The 'work_pc_pwd' MUST be 8+ chars with 3 of 4 character types.<|im_end|>\n"
            "<|im_start|>assistant\n{"
        )

        try:
            # Correctly indented block
            pipe.generate(prompt, max_new_tokens=1024, streamer=streamer, do_sample=True, temperature=0.8)
            
            data = repair_json_light(full_response)
            
            if data and check_password_complexity(data.get("work_pc_pwd", "")):
                results.append(data)
                print(f"[{len(results)}/{total_count}] Generated: {data.get('name')}")
                
                # Save progress every 2 items
                if len(results) % 2 == 0:
                    with open(output_file, "w") as f:
                        json.dump(results, f, indent=4)
            else:
                print(f"Attempt failed validation or JSON was malformed. Retrying...")
                
        except Exception as e:
            print(f"Inference error: {e}")
            time.sleep(1)

    print(f"\nDONE! {len(results)} personas saved to {output_file}")

if __name__ == "__main__":
    run_npu_persona_batch(100)
