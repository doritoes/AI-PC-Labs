import os
import json
import re
import string
import time
import openvino_genai as ov_genai

# --- 1. HARDWARE CONFIG (Verified for v4515 Drivers) ---
os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"

def repair_json_light(raw_text):
    """Fixes common LLM JSON errors like trailing commas or missing braces."""
    try:
        # 1. Extract the core JSON block
        match = re.search(r'(\{.*\})', raw_text.replace('\n', ' '), re.DOTALL)
        if not match:
            return None
        json_str = match.group(1).strip()
        
        # 2. Fix trailing commas: {"a": 1, } -> {"a": 1}
        json_str = re.sub(r',\s*\}', '}', json_str)
        json_str = re.sub(r',\s*\]', ']', json_str)
        
        return json.loads(json_str)
    except Exception:
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
    
    # EXACT config from your working test script
    pipeline_config = {
        "MAX_PROMPT_LEN": 1024,
        "MIN_RESPONSE_LEN": 128,
        "PREFILL_HINT": "STATIC",
        "GENERATE_HINT": "BEST_PERF",
        "PERFORMANCE_HINT": "LATENCY",
        "CACHE_DIR": cache_dir
    }

    print(f"--- INITIALIZING NPU ({total_count} Identités) ---")
    try:
        pipe = ov_genai.LLMPipeline(model_path, "NPU", **pipeline_config)
    except Exception as e:
        print(f"FAILED TO LOAD NPU: {e}")
        return

    results = []
    i = 0
    while len(results) < total_count:
        i += 1
        full_response = "{" # Force-starting the JSON object
        
        def streamer(subword):
            nonlocal full_response
            full_response += subword
            return ov_genai.StreamingStatus.RUNNING

        # Shorter prompt helps NPU keep the JSON structure intact
        prompt = (
            "<|im_start|>system\nYou are a JSON generator. No chat.<|im_end|>\n"
            "<|im_start|>user\nGenerate one persona: {name, city, job, "
            "work_pwd (8+ chars, 3 of 4 types)}. Return JSON only.<|im_end|>\n"
            "<|im_start|>assistant\n{"
        )

        try:
            # We use do_sample=True
