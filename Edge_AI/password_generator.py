import os
import json
import csv
import re
import string
import openvino_genai as ov_genai

# Mandatory for Intel NPU stability
os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"

def check_password_complexity(pwd):
    """Validation helper for the '3 of 4' rule."""
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
    pipe = ov_genai.LLMPipeline(model_path, "NPU")
    results = []

    for i in range(count):
        valid_entry = False
        attempts = 0
        
        while not valid_entry and attempts < 5:
            # Note: We use the system prompt to explicitly define the 3/4 rule
            prompt = "<|im_start|>user\nGenerate a unique persona and a Work PC password (8+ chars, 3 of 4: Upper, Lower, Num, Special). Return JSON.<|im_end|>\n<|im_start|>assistant\n"
            response = pipe.generate(prompt, max_new_tokens=512)
            
            try:
                # Basic JSON extraction
                data = json.loads(re.search(r'({.*})', response.replace('\n', ''), re.DOTALL).group(1))
                work_pwd = data.get("password_work", "")
                
                if check_password_complexity(work_pwd):
                    results.append(data)
                    valid_entry = True
                    print(f"[{i+1}/{count}] Valid Persona Generated.")
                else:
                    attempts += 1
                    print(f"[{i+1}/{count}] Password failed complexity. Retrying...")
            except:
                attempts += 1

    # Save final results
    with open("validated_identities.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_validated_batch(5) # Testing with 5 first
