import json
import csv
import re
import time
import random
from openai import OpenAI

# Configuration
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
TOTAL_PERSONAS = 100
MAX_RETRIES = 3  # How many times to retry a single persona if it fails

def get_ai_response(attempt_number):
    """Refined prompt to ensure all your specific fields are included."""
    # We add a random 'seed' word to the prompt to force the local model 
    # to stay creative across 100 iterations.
    seeds = ["adventurous", "quiet", "organized", "forgetful", "nostalgic", "technical"]
    seed = random.choice(seeds)
    
    prompt = f"""
    Create a unique identity for a {seed} human aged 25-65. 
    Output ONLY a valid JSON object wrapped in ```json ``` blocks.
    
    Include these exact keys:
    "name", "city", "birthdate", "zodiac", "occupation", "address", 
    "transportation", "blood_type", "hair_color", "eye_color", 
    "height", "weight", "religion", "fav_vacation_spot", "fav_season", 
    "fav_animal", "best_friend", "birthplace",
    "password_personal_email", "password_work_pc", "password_banking",
    "logic_explanation"
    
    Note: Passwords must feel 'human'—derived from the persona's life. 
    Work PC must be 8+ chars with 3 of 4 classes (Upper, Lower, Number, Symbol).
    """

    response = client.chat.completions.create(
        model="qwen",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9  # High creativity
    )
    return response.choices[0].message.content

def extract_json(text):
    """Cleanly extracts JSON from the AI's markdown response."""
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    return json.loads(text) # Fallback if no markdown used

def main():
    final_results = []
    
    for i in range(1, TOTAL_PERSONAS + 1):
        success = False
        retries = 0
        
        while not success and retries < MAX_RETRIES:
            try:
                print(f"Persona {i}/{TOTAL_PERSONAS} (Attempt {retries + 1})...", end="\r")
                
                raw_output = get_ai_response(retries)
                data = extract_json(raw_output)
                
                # Basic validation to ensure passwords exist
                if "password_banking" in data:
                    final_results.append(data)
                    success = True
                
            except Exception as e:
                retries += 1
                wait_time = retries * 2 # Wait 2s, then 4s, etc.
                print(f"\n[!] Error on persona {i}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        if not success:
            print(f"\n[X] Failed to generate persona {i} after {MAX_RETRIES} attempts. Skipping.")

    # Save Results
    if final_results:
        # Save JSON
        with open("personas.json", "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=4)
        
        # Save CSV
        keys = final_results[0].keys()
        with open("personas.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(final_results)
            
        print(f"\n\nDone! Successfully generated {len(final_results)} personas.")

if __name__ == "__main__":
    main()
