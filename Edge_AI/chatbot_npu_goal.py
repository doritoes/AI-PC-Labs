import os
import time
import openvino_genai as ov_genai

def format_prompt(user_query):
    """
    Wraps the user input in the Qwen2.5 'ChatML' template.
    This gives the model a persona and prevents 'hallucinations'.
    """
    system_message = "You are a helpful AI Developer Tutor. Explain concepts simply and use step-by-step logic."
    return f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"

def run_npu_chatbot():
    # 1. ENVIRONMENT & DIRECTORY SETUP
    os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"
    model_path = os.path.join(os.environ['USERPROFILE'], 'Edge-AI', 'models', 'qwen-1.5b')

    # Ensure cache directory exists for instant loading
    cache_dir = os.path.join(os.getcwd(), 'npu_cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    print("\n" + "="*50)
    print("  INTEL NPU TUTOR BOT (Qwen 2.5)")
    print("="*50)
    print("INITIALIZING: Mapping NPU pathways...")

    # 2. CONFIGURATION (Kwargs + Static Constraints)
    pipeline_config = {
        "MAX_PROMPT_LEN": 1024,
        "MIN_RESPONSE_LEN": 128,
        "PREFILL_HINT": "STATIC",
        "GENERATE_HINT": "BEST_PERF",
        "PERFORMANCE_HINT": "LATENCY",
        "CACHE_DIR": cache_dir
    }

    try:
        pipe = ov_genai.LLMPipeline(model_path, "NPU", **pipeline_config)
    except Exception as e:
        print(f"\n[CRITICAL ERROR]: {e}")
        return

    print("\n" + "*"*50)
    print("   SUCCESS: NPU online")
    print("   Type 'quit' to exit the program")
    print("*"*50)

    # 3. CHAT LOOP
    while True:
        user_input = input("\nUser: ")

        if user_input.lower() in ['quit', 'exit']:
            print("Shutting down NPU sessions... Goodbye!")
            break

        if not user_input.strip():
            continue

        # Performance Tracking
        start_time = time.time()
        first_token_time = None
        token_count = 0

        # Apply the Chat Template
        formatted_input = format_prompt(user_input)

        print("NPU Tutor: ", end="", flush=True)

        def streamer(subword):
            nonlocal first_token_time, token_count
            if first_token_time is None:
                first_token_time = time.time()
            print(subword, end="", flush=True)
            token_count += 1
            return ov_genai.StreamingStatus.RUNNING

        try:
            # We use do_sample=False for consistent, logical answers (Greedy Decoding)
            pipe.generate(formatted_input, max_new_tokens=256, streamer=streamer, do_sample=False)

            # Show Performance Metrics
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
