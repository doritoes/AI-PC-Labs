import openvino_genai as ov_genai
import time
import os

def run_npu_chatbot():
    # 1. HARDWARE ENVIRONMENT FIX
    os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"
    
    # 2. PATH SETUP
    model_path = os.path.join(os.environ['USERPROFILE'], 'Edge-AI', 'models', 'qwen-1.5b')

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return

    # 3. INITIALIZE NPU (Final Stability Config)
    print("\n--- Initializing Intel NPU (Stability Mode) ---")
    
    # We use 'YES'/'NO' strings which are the standard for OpenVINO NPU config
    config = {
        "NPU_RUN_INFERENCES_SEQUENTIALLY": "NO",
        "PERFORMANCE_HINT": "LATENCY",
        # Disabling NPUW prevents the 'MatMul map' error by using the standard NPU path
        "NPU_USE_NPUW": "NO" 
    }

    try:
        pipe = ov_genai.LLMPipeline(model_path, "NPU", **config)
    except Exception as e:
        print(f"\n[CRITICAL ERROR]: {e}")
        return

    # 4. CHAT INTERFACE
    print("\n" + "="*50)
    print("NPU CHATBOT ONLINE (Type 'quit' to stop)")
    print("="*50)

    while True:
        prompt = input("\nStudent: ")
        if prompt.lower() in ['quit', 'exit']: break
        
        start_time = time.time()
        first_token_time = None
        token_count = 0

        print("NPU Assistant: ", end="", flush=True)

        def streamer(subword):
            nonlocal first_token_time, token_count
            if first_token_time is None:
                first_token_time = time.time()
            print(subword, end="", flush=True)
            token_count += 1
            return ov_genai.StreamingStatus.RUNNING

        pipe.generate(prompt, max_new_tokens=256, streamer=streamer)

        # 5. METRICS
        end_time = time.time()
        ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
        gen_duration = end_time - first_token_time if first_token_time else 0
        tps = token_count / gen_duration if gen_duration > 0 else 0

        print(f"\n\n" + "-"*40)
        print(f"NPU PERFORMANCE METRICS:")
        print(f">> TTFT: {ttft:.2f} ms | TPS: {tps:.2f} tok/s")
        print(f"-"*40)

if __name__ == "__main__":
    run_npu_chatbot()
