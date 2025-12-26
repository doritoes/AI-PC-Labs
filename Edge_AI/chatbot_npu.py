import openvino_genai as ov_genai
import time
import os
import shutil

def run_npu_chatbot():
    # 1. HARDWARE ENVIRONMENT STABILITY
    # Helps prevent memory allocation issues on current NPU driver stacks
    os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"
    
    # 2. PATH SETUP
    model_path = os.path.join(os.environ['USERPROFILE'], 'Edge-AI', 'models', 'qwen-1.5b')
    cache_path = os.path.join(os.environ['LOCALAPPDATA'], 'OpenVINO', 'cache')

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return

    # 3. INITIALIZE NPU (Optimized for v4515 Driver)
    print("\n" + "-"*50)
    print("INITIALIZING INTEL NPU (v4515 DETECTED)")
    print(f"MODEL: Qwen2.5-1.5B-Instruct (INT4)")
    print("-"*50)
    
    # PERFORMANCE_HINT: LATENCY ensures the NPU prioritizes immediate response
    config = {
        "PERFORMANCE_HINT": "LATENCY"
    }

    try:
        # Initial compilation may take 30-45 seconds on the new driver
        pipe = ov_genai.LLMPipeline(model_path, "NPU", **config)
    except Exception as e:
        print(f"\n[CRITICAL ERROR]: {e}")
        return

    # 4. CHAT INTERFACE
    print("\n" + "="*50)
    print("NPU CHATBOT ONLINE (Type 'quit' to exit)")
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
            return ov_genai.StreamingStatus.RUNNING

        # 6. GENERATION
        try:
            # We use max_new_tokens=256 to stay within the driver's optimal static buffer
            pipe.generate(prompt, max_new_tokens=256, streamer=streamer)
        except Exception as e:
            print(f"\n[GENERATION ERROR]: {e}")
            break

        # 7. METRIC CALCULATION
        end_time = time.time()
        
        # Time to First Token (Responsiveness)
        ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
        
        # Tokens Per Second (Throughput)
        gen_duration = end_time - first_token_time if first_token_time else 0
        tps = token_count / gen_duration if gen_duration > 0 else 0

        print(f"\n\n" + "-"*40)
        print(f"NPU PERFORMANCE METRICS:")
        print(f">> TTFT (Responsiveness): {ttft:.2f} ms")
        print(f">> TPS (Throughput):      {tps:.2f} tokens/sec")
        print(f"-"*40)

if __name__ == "__main__":
    run_npu_chatbot()
