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

def parse_persona_text(text):
    """Extracts data from natural language if JSON fails."""
    # Look for patterns like Name: [Name], Email Pwd: [Pwd], etc.
    name = re.search(r"Name:\s*(.*)", text)
    logic = re.search(r"Logic:\s*(.*)", text)
    email = re.search(r"Email Pwd:\s*(\S+)", text)
    work = re.search(r"Work Pwd:\s*(\S+)", text)
    
    if name and email and work:
        return {
            "name": name.group(1).strip(),
            "logic_note": logic.group(1).strip() if logic else "N/A",
            "email_pwd": email.group(1).strip(),
            "work_pwd": work.group(1).strip()
        }
    return None

def run_npu_human_study(total_count=100):
    print("--- INITIALIZING NPU ---")
    pipe = ov_genai.LLMPipeline(MODEL_PATH, "NPU", **NPU_CONFIG)

    results = []
    attempts = 0
    start_time = time.time()
    
    jobs = ["Chef", "Pilot", "Nurse", "Architect", "Scientist", "Teacher"]

    print(f"--- STARTING ROBUST TEXT-PARSED STUDY: {total_count} SAMPLES ---")

    while len(results) < total_count:
        attempts += 1
        job = random.choice(jobs)
        dob = generate_random_birthdate()
        
        full_response = ""
        def streamer(subword):
            nonlocal full_response
            full_response += subword
            return ov_genai.StreamingStatus.RUNNING

        # TEXT-BASED PROMPT: Easier for NPU than JSON
        prompt = (
            f"<|im_start|>system\nYou are a profiler. Speak naturally. Use this format:\n"
            f"Name: [Full Name]\nLogic: [Why they chose these]\nEmail Pwd: [Hobby password]\nWork Pwd: [Career password]<|im_end|>\n"
            f"<|im_start|>user\nProfile a {job} born {dob}.<|im_end|>\n"
            f"<|im_start|>assistant\nName:"
        )

        try:
            eff = (len(results) / attempts) * 100 if attempts > 0 else 0
            print(f"[{len(results)+1}/{total_count}] Eff: {eff:.1f}% | Mode: {job}...".ljust(65), end="\r")
            
            # Shorter token limit to prevent infinite loops
            pipe.generate(prompt, max_new_tokens=200, streamer=streamer, do_sample=True, temperature=0.8)
            
            # Attempt to parse the text output
            data = parse_persona_text("Name:" + full_response)
            
            if data:
                # Basic 'Human' check (No QWERTY)
                if not any(x in data["work_pwd"].lower() for x in ["qwerty", "asdf"]):
                    data["job"], data["dob"] = job, dob
                    results.append(data)
                    
                    print(f"[{len(results)}/{total_count}] ✅ {data['name'][:12].ljust(12)} | Work: {data['work_pwd']}")
                    print(f"      Logic: {data['logic_note'][:55]}...")
                    
                    if len(results) % 5 == 0:
                        with open("robust_human_study.json", "w") as f:
                            json.dump(results, f, indent=4)
        except Exception as e:
            continue

    print(f"\nSUCCESS: 100 Samples in {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    run_npu_human_study(100)
