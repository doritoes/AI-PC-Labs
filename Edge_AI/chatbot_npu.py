import openvino_genai as ov_genai
import time
import os

def run_npu_chatbot():
    # 1. PATH TO YOUR CONVERTED MODEL
    model_path = os.path.join(os.environ['USERPROFILE'], 'Edge-AI', 'models', 'phi-3-mini-int4')
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}. Please check your download.")
        return

    # 2. NPU INITIALIZATION
    print("Loading model to NPU... (This may take up to 30 seconds for first-time compilation)")
    
    # We use GENERATE_HINT: BEST_PERF to optimize the NPU's throughput
    config = {
        "GENERATE_HINT": "BEST_PERF",
        "PERFORMANCE_HINT": "LATENCY"
    }
    
    # The LLMPipeline handles tokenization and execution on the NPU
    print("--- Initializing NPU (Safe Mode) ---")
    pipe = ov_genai.LLMPipeline(model_path, "NPU")
    #pipe = ov_genai.LLMPipeline(model_path, "NPU", **config)

    print("\n" + "="*50)
    print("NPU CHATBOT ONLINE (Type 'quit' to stop)")
    print("="*50)

    while True:
        prompt = input("\nStudent: ")
        if prompt.lower() in ['quit', 'exit']: break

        # 3. METRICS TRACKING
        start_time = time.time()
        first_token_time = None
        token_count = 0

        print("NPU Assistant: ", end="", flush=True)

        # 4. STREAMER FUNCTION
        def streamer(subword):
            nonlocal first_token_time, token_count
            # Catch the exact moment the first token appears
            if first_token_time is None:
                first_token_time = time.time()
            
            print(subword, end="", flush=True)
            token_count += 1
            return ov_genai.StreamingStatus.RUNNING

        # 5. GENERATION
        pipe.generate(prompt, max_new_tokens=256, streamer=streamer)

        # 6. CALCULATE RESULTS
        end_time = time.time()
        
        # TTFT: Time from 'Enter' key to the first word appearing
        ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
        
        # TPS: The generation speed (tokens per second)
        generation_duration = end_time - first_token_time if first_token_time else 0
        tps = token_count / generation_duration if generation_duration > 0 else 0

        print(f"\n\n" + "-"*30)
        print(f"NPU METRICS:")
        print(f">> Time to First Token (TTFT): {ttft:.2f} ms")
        print(f">> Tokens Per Second (TPS):    {tps:.2f} tok/s")
        print(f"-"*30)

if __name__ == "__main__":
    run_npu_chatbot()
