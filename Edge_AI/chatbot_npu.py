import openvino_genai as ov_genai
import time
import os
import shutil

def run_npu_chatbot():
    # 1. HARDWARE ENVIRONMENT FIXES (Automatic)
    # This bypasses the L0 memory allocation error we saw earlier
    os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"
    
    # 2. PATH SETUP
    # Updated to the Qwen-1.5b directory
    model_path = os.path.join(os.environ['USERPROFILE'], 'Edge-AI', 'models', 'qwen-1.5b')
    cache_path = os.path.join(os.environ['LOCALAPPDATA'], 'OpenVINO', 'cache')

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return

    # 3. INITIALIZE NPU WITH COMPATIBILITY FLAGS
    print("\n--- Initializing Intel NPU (Full Compatibility Mode) ---")
    
    # We are adding a 'DUMP' and 'NO_COMPILATION' hint to force a cleaner 
    # mapping of the MatMul layers that were previously failing.
    config = {
        "NPU_RUN_INFERENCES_SEQUENTIALLY": "OFF",
        "PERFORMANCE_HINT": "LATENCY",
        "NPU_USE_NPUW": "NO"  # This forces OpenVINO to use the standard NPU driver
                               # instead of the "Weights" optimization wrapper 
                               # which is currently causing the MatMul map error.
    }
    try:
        # Loading the pipeline. First run will trigger compilation.
        pipe = ov_genai.LLMPipeline(model_path, "NPU", **config)
    except Exception as e:
        print(f"\n[CRITICAL ERROR]: {e}")
        if os.path.exists(cache_path):
            print("Cleaning cache and recommending a restart...")
            shutil.rmtree(cache_path)
        return

    # 4. CHAT INTERFACE
    print("\n" + "="*50)
    print("NPU CHATBOT ONLINE (Type 'quit' to stop)")
    print("="*50)

    while True:
        prompt = input("\nStudent: ")
        if prompt.lower() in ['quit', 'exit']: break
        if not prompt.strip(): continue

        # Metric Tracking
        start_time = time.time()
        first_token_time = None
        token_count = 0

        print("NPU Assistant: ", end="", flush=True)

        # 5. STREAMER
        def streamer(subword):
            nonlocal first_token_time, token_count
            if first_token_time is None:
                first_token_time = time.time()
            print(subword, end="", flush=True)
            token_count += 1
            # Return RUNNING to keep generating
            return ov_genai.StreamingStatus.RUNNING

        # 6. GENERATION
        try:
            # max_new_tokens limits the response length to save NPU resources
            pipe.generate(prompt, max_new_tokens=256, streamer=streamer)
        except Exception as e:
            print(f"\n[GENERATION ERROR]: {e}")
            break

        # 7. METRIC CALCULATION
        end_time = time.time()
        
        # Time to First Token (TTFT)
        ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
        
        # Tokens Per Second (TPS)
        gen_duration = end_time - first_token_time if first_token_time else 0
        tps = token_count / gen_duration if gen_duration > 0 else 0

        print(f"\n\n" + "-"*40)
        print(f"NPU PERFORMANCE METRICS:")
        print(f">> TTFT (Responsiveness): {ttft:.2f} ms")
        print(f">> TPS (Throughput):      {tps:.2f} tokens/sec")
        print(f"-"*40)

if __name__ == "__main__":
    run_npu_chatbot()
