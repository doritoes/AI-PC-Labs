import openvino_genai as ov_genai
import time
import os

def run_npu_chatbot():
    # Final stability flag for Intel NPU drivers
    os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"
    
    model_path = os.path.join(os.environ['USERPROFILE'], 'Edge-AI', 'models', 'qwen-1.5b')

    print("\n" + "="*50)
    print("  INTEL NPU ACCELERATED CHATBOT (v4515)  ")
    print("="*50)
    print("INITIALIZING: Please wait while NPU buffers are reserved...")

    # Static container configuration for v4515 stability
    pipeline_config = {
        "MAX_PROMPT_LEN": 1024,
        "MIN_RESPONSE_LEN": 128,
        "PREFILL_HINT": "STATIC",
        "GENERATE_HINT": "BEST_PERF",
        "PERFORMANCE_HINT": "LATENCY"
    }

    try:
        pipe = ov_genai.LLMPipeline(model_path, "NPU", pipeline_config)
    except Exception as e:
        print(f"\n[CRITICAL ERROR]: {e}")
        return

    print("\n" + "*"*50)
    print("   SUCCESS: NPU ONLINE   ")
    print("   Type 'quit' to exit the program   ")
    print("*"*50)

    while True:
        prompt = input("\nStudent: ")
        
        # Check for quit command
        if prompt.lower() in ['quit', 'exit']:
            print("Shutting down NPU sessions... Goodbye!")
            break
        
        if not prompt.strip(): continue

        # Performance Tracking
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

        try:
            # Generate using the NPU
            pipe.generate(prompt, max_new_tokens=256, streamer=streamer, do_sample=False)
            
            # End of response metrics
            end_time = time.time()
            if first_token_time:
                ttft = (first_token_time - start_time) * 1000
                gen_duration = end_time - first_token_time
                tps = token_count / gen_duration if gen_duration > 0 else 0
                
                print(f"\n\n[METRICS] TTFT: {ttft:.1f}ms | Speed: {tps:.1f} tokens/sec")
            
        except Exception as e:
            print(f"\n[GENERATION ERROR]: {e}")
            break

if __name__ == "__main__":
    run_npu_chatbot()
