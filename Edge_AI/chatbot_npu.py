import openvino_genai as ov_genai
import time
import os
import shutil

def run_npu_chatbot():
    # 1. HARDWARE ENVIRONMENT FIXES
    # Force the L0 memory bypass to prevent 'Out of Host Memory' errors
    os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"
    
    # 2. PATH SETUP
    model_path = os.path.join(os.environ['USERPROFILE'], 'Edge-AI', 'models', 'phi-3-mini-int4')
    cache_path = os.path.join(os.environ['LOCALAPPDATA'], 'OpenVINO', 'cache')

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please run the 'hf download' command first.")
        return

    # 3. INITIALIZE NPU WITH STREAMING SETTINGS
    print("\n--- Initializing Intel NPU (AI PC Accelerator) ---")
    print("Step 1: Configuring hardware streaming lanes...")
    
    # NPUW_CONF: RUN_INFERENCES_SEQUENTIALLY=OFF is required for streaming on many drivers
    # PERFORMANCE_HINT: LATENCY ensures the fastest response time for the user
    config = {
        "NPUW_CONF": "RUN_INFERENCES_SEQUENTIALLY=OFF",
        "PERFORMANCE_HINT": "LATENCY"
    }

    try:
        # Loading the pipeline
        pipe = ov_genai.LLMPipeline(model_path, "NPU", **config)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] NPU Initialization failed: {e}")
        print("\nAttempting to clear cache and restart...")
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
        print("Please run the script one more time.")
        return

    # 4. CHAT LOOP
    print("\n" + "="*50)
    print("NPU CHATBOT ONLINE (Type 'quit' to exit)")
    print("="*50)

    while True:
        prompt = input("\nStudent: ")
        if prompt.lower() in ['quit', 'exit']: 
            break
        if not prompt.strip():
            continue

        # Tracking metrics
        start_time = time.time()
        first_token_time = None
        token_count = 0

        print("NPU Assistant: ", end="", flush=True)

        # Streamer function to capture tokens and timing
        def streamer(subword):
            nonlocal first_token_time, token_count
            if first_token_time is None:
                first_token_time = time.time()
            
            print(subword, end="", flush=True)
            token_count += 1
            # Signal to keep generating
            return ov_genai.StreamingStatus.RUNNING

        # 5. GENERATE RESPONSE
        try:
            pipe.generate(prompt, max_new_tokens=256, streamer=streamer)
        except Exception as e:
            print(f"\n[GENERATION ERROR]: {e}")
            break

        # 6. CALCULATE & DISPLAY METRICS
        end_time = time.time()
        
        # TTFT (Time to First Token)
        ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
        
        # TPS (Tokens Per Second)
        gen_duration = end_time - first_token_time if first_token_time else 0
        tps = token_count / gen_duration if gen_duration > 0 else 0

        print(f"\n\n" + "-"*40)
        print(f"LAB METRICS:")
        print(f">> Time to First Token (TTFT): {ttft:.2f} ms")
        print(f">> Generation Speed (TPS):     {tps:.2f} tokens/sec")
        print(f"-"*40)

if __name__ == "__main__":
    run_npu_chatbot()
