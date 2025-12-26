import openvino_genai as ov_genai
import time
import os
import shutil

def run_npu_chatbot():
    # 1. HARDWARE ENVIRONMENT FIXES
    # Force the L0 memory bypass to prevent 'Out of Host Memory' errors on some systems
    os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"
    
    # 2. PATH SETUP
    model_path = os.path.join(os.environ['USERPROFILE'], 'Edge-AI', 'models', 'phi-3-mini-int4')
    cache_path = os.path.join(os.environ['LOCALAPPDATA'], 'OpenVINO', 'cache')

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return

    # 3. INITIALIZE NPU WITH MODERN CONFIGURATION
    print("\n--- Initializing Intel NPU (AI PC Accelerator) ---")
    
    # In newer 2025 drivers, settings are passed directly to the NPU
    # NPU_RUN_INFERENCES_SEQUENTIALLY=OFF enables smooth token streaming
    config = {
        "NPU_RUN_INFERENCES_SEQUENTIALLY": "OFF",
        "PERFORMANCE_HINT": "LATENCY"
    }

    try:
        # The LLMPipeline handles tokenization and NPU execution
        pipe = ov_genai.LLMPipeline(model_path, "NPU", **config)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] NPU Initialization failed: {e}")
        # Auto-clear cache if a mismatch is detected
        if os.path.exists(cache_path):
            print("Clearing corrupted NPU cache...")
            shutil.rmtree(cache_path)
        return

    # 4. CHAT INTERFACE
    print("\n" + "="*50)
    print("NPU CHATBOT ONLINE (Type 'quit' to stop)")
    print("="*50)

    while True:
        prompt = input("\nStudent: ")
        if prompt.lower() in ['quit', 'exit']: 
            break
        if not prompt.strip():
            continue

        # Initialize metric trackers
        start_time = time.time()
        first_token_time = None
        token_count = 0

        print("NPU Assistant: ", end="", flush=True)

        # 5. STREAMER FUNCTION
        def streamer(subword):
            nonlocal first_token_time, token_count
            if first_token_time is None:
                first_token_time = time.time()
            
            print(subword, end="", flush=True)
            token_count += 1
            return ov_genai.StreamingStatus.RUNNING

        # 6. GENERATION
        try:
            pipe.generate(prompt, max_new_tokens=256, streamer=streamer)
        except Exception as e:
            print(f"\n[GENERATION ERROR]: {e}")
            break

        # 7. METRIC CALCULATION
        end_time = time.time()
        
        # TTFT: How fast the NPU "prefills" the prompt and starts talking
        ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
        
        # TPS: The generation throughput once it starts
        gen_duration = end_time - first_token_time if first_token_time else 0
        tps = token_count / gen_duration if gen_duration > 0 else 0

        print(f"\n\n" + "-"*40)
        print(f"NPU PERFORMANCE METRICS:")
        print(f">> Time to First Token (TTFT): {ttft:.2f} ms")
        print(f">> Tokens Per Second (TPS):    {tps:.2f} tok/s")
        print(f"-"*40)

if __name__ == "__main__":
    run_npu_chatbot()
