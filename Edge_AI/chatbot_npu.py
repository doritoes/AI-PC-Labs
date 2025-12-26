import openvino_genai as ov_genai
import time
import os

def run_npu_chatbot():
    # 1. HARDWARE ENVIRONMENT FIX
    # This environment variable tells the NPU driver to use a more stable 
    # memory allocation mode, preventing 'Silent Crises' and crashes.
    os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"
    
    # 2. PATH SETUP 
    model_path = os.path.join(os.environ['USERPROFILE'], 'Edge-AI', 'models', 'tiny-llama')

    if not os.path.exists(model_path):
        print(f"ERROR: TinyLlama model not found at {model_path}")
        return

    # 3. INITIALIZE NPU (Stability Configuration)
    print("\n" + "-"*50)
    print("INITIALIZING INTEL NPU (AI PC ACCELERATOR)")
    print("MODEL: TinyLlama-1.1B (INT4 Quantized)")
    print("-"*50)
    
    # NPU_USE_NPUW: 'NO' bypasses the experimental weights wrapper to ensure 
    # the MatMul mapping error from earlier is avoided.
    config = {
        "NPU_USE_NPUW": "NO",
        "PERFORMANCE_HINT": "LATENCY"
    }

    try:
        # Load the pipeline. The first run takes a moment to compile.
        pipe = ov_genai.LLMPipeline(model_path, "NPU", **config)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] NPU Initialization failed: {e}")
        return

    # 4. CHAT LOOP
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

        # 7. PERFORMANCE METRICS
        end_time = time.time()
        ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
        gen_duration = end_time - first_token_time if first_token_time else 0
        tps = token_count / gen_duration if gen_duration > 0 else 0

        print(f"\n\n" + "-"*40)
        print(f"NPU LAB METRICS:")
        print(f">> Responsiveness (TTFT): {ttft:.2f} ms")
        print(f">> Speed (TPS):           {tps:.2f} tokens/sec")
        print(f"-"*40)

if __name__ == "__main__":
    run_npu_chatbot()
