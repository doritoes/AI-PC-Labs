import json
import re
import random
import time
from datetime import datetime, timedelta
import openvino_genai as ov_genai
from config import MODEL_PATH, NPU_CONFIG

def generate_random_birthdate():
    days_ago = random.randint(365*20, 365*65)
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

def run_npu_human_study(total_count=100):
    print("--- INITIALIZING NPU ---")
    pipe = ov_genai.LLMPipeline(MODEL_PATH, "NPU", **NPU_CONFIG)

    results = []
    attempts = 0
    start_time = time.time()
    
    locs = ["London", "Tokyo", "Berlin", "New York", "Paris", "Sydney", "Mumbai", "Seoul"]
    jobs = ["Chef", "Pilot", "Nurse", "Architect", "Scientist", "Teacher", "Lawyer"]

    print(f"--- RESTORING HUMAN BEHAVIOR STUDY: {total_count} SAMPLES ---")

    while len(results) < total_count:
        attempts += 1
        loc, job = random.choice(locs), random.choice(jobs)
        dob = generate_random_birthdate()
        
        full_response = ""
        def streamer(subword):
            nonlocal full_response
            full_response += subword
            return ov_genai.StreamingStatus.RUNNING

        # This is the "Sophie Dupuy" Logic: Human reasoning first.
        prompt = (
            f"<|im_start|>system\nYou are a psychological profiler. Output ONLY JSON.\n"
            f"Humans use simple memories for personal mail and professional tools for work.<|im_end|>\n"
            f"<|im_start|>user\nCreate a persona for a {job} in {loc}.\n"
            f"Requirements:\n"
            f"1. A name and a logic_note explaining their password habits.\n"
            f"2. email_pwd (lazy/hobby-based).\n"
            f"3. work_pwd (leetspeak/complex based on their career).\n"
            f"JSON format: {{'name', 'logic_note', 'email_pwd', 'work_pwd'}}<|im_end|>\n"
            f"<|im_start|>assistant\n{{\"name\":"
        )

        try:
            eff = (len(results) / attempts) * 100 if attempts > 0 else 0
            print(f"[{len(results)+1}/{total_count}] Eff: {eff:.1f}% | Profiling {job}...".ljust(65), end="\r")
            
            pipe.generate(prompt, max_new_tokens=450, streamer=streamer, do_sample=True, temperature=0.8)
            
            # Reconstruct JSON with the prefilled name key
            raw = "{\"name\":" + full_response.strip()
            if not raw.endswith("}"): raw += "}"
            
            match = re.search(r'(\{.*\})', raw.replace('\n', ' '), re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                
                # Filter for quality: Must have the logic note and reasonable passwords
                if all(k in data for k in ["name", "logic_note", "email_pwd", "work_pwd"]):
                    wp = data["work_pwd"].lower()
                    if not any(x in wp for x in ["qwerty", "asdf", "123456"]):
                        data["job"], data["city"], data["dob"] = job, loc, dob
                        results.append(data)
                        
                        # High-visibility logging
                        print(f"[{len(results)}/{total_count}] ✅ {data['name'][:12].ljust(12)} | Note: {data['logic_note'][:45]}...")
                        print(f"      Email: {data['email_pwd'].ljust(15)} | Work: {data['work_pwd']}")
                        
                        if len(results) % 5 == 0:
                            with open("human_study_restored.json", "w") as f:
                                json.dump(results, f, indent=4)
        except Exception:
            continue

    print(f"\nSUCCESS: 100 Behavioral Samples in {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    run_npu_human_study(100)
