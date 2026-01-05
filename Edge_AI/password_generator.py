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

def run_npu_study(total_count=100):
    print("--- INITIALIZING NPU ---")
    try:
        pipe = ov_genai.LLMPipeline(MODEL_PATH, "NPU", **NPU_CONFIG)
    except Exception as e:
        print(f"Hardware Error: {e}")
        return

    results = []
    attempts = 0
    start_time = time.time()
    
    # Study Variables
    locs = ["London", "Tokyo", "Berlin", "New York", "Paris", "Sydney"]
    jobs = ["Chef", "Pilot", "Nurse", "Architect", "Scientist", "Teacher"]

    print(f"--- STARTING BATCH: {total_count} SAMPLES ---")

    while len(results) < total_count:
        attempts += 1
        loc, job = random.choice(locs), random.choice(jobs)
        dob = generate_random_birthdate()
        
        full_response = ""
        def streamer(subword):
            nonlocal full_response
            full_response += subword
            return ov_genai.StreamingStatus.RUNNING

        # FEW-SHOT PROMPT: Showing the model exactly how a human thinks
        prompt = (
            f"<|im_start|>system\nYou are a profile generator. Output ONLY JSON.\n"
            f"Example:\n"
            f"User: Architect in London.\n"
            f"Assistant: {{\"name\": \"James Miller\", \"logic\": \"Loves cycling, but uses CAD tools at work.\", \"email_pwd\": \"RedBike88\", \"work_pwd\": \"Sk3tchUp!2024\"}}<|im_end|>\n"
            f"<|im_start|>user\nCreate a persona for a {job} in {loc}.<|im_end|>\n"
            f"<|im_start|>assistant\n{{\"name\":"
        )

        try:
            eff = (len(results) / attempts) * 100 if attempts > 0 else 0
            print(f"[{len(results)+1}/{total_count}] Eff: {eff:.1f}% | Task: {job}...".ljust(60), end="\r")
            
            # Use temperature 0.7 for a balance of creativity and structure
            pipe.generate(prompt, max_new_tokens=256, streamer=streamer, do_sample=True, temperature=0.7)
            
            # Parse and clean
            raw = "{\"name\":" + full_response.strip()
            if not raw.endswith("}"): raw += "}"
            
            # Find the JSON block
            match = re.search(r'(\{.*\})', raw.replace('\n', ' '), re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                
                # Validation: Ensure all 3 human fields exist
                if all(k in data for k in ["name", "email_pwd", "work_pwd"]):
                    data["job"], data["city"], data["dob"] = job, loc, dob
                    results.append(data)
                    
                    # Log the human-like result
                    print(f"[{len(results)}/{total_count}] ✅ {data['name'][:12].ljust(12)} | Email: {data['email_pwd'].ljust(12)} | Work: {data['work_pwd']}")
                    
                    if len(results) % 5 == 0:
                        with open("human_passwords_final.json", "w") as f:
                            json.dump(results, f, indent=4)
        except Exception:
            continue

    print(f"\nSUCCESS: 100 personas saved in {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    run_npu_study(100)
