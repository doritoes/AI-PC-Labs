import openvino_genai as ov_genai
import time
import os

def format_prompt(user_query):
    system_message = "You are a precise AI Assistant. Provide concise, logical answers."
    return f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"

def run_automated_test():
    # 1. SETUP
    os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"
    model_path = os.path.join(os.environ['USERPROFILE'], 'Edge-AI', 'models', 'qwen-1.5b')
    cache_dir = os.path.join(os.getcwd(), 'npu_cache')
    report_file = "npu_test_results.md"

    test_questions = [
        "What is the square root of 144?",
        "If all bloops are blips and all blips are blops, are all bloops blops?",
        "Explain photosynthesis in one sentence.",
        "Write a 4-line poem about a computer chip.",
        "How many legs does a spider have?"
    ]

    print(f"\nInitializing NPU for {len(test_questions)} automated tests...")

    # FULL STABILITY BLOCK - This matches your working manual script
    config = {
        "NPU_USE_NPUW": "NO",
        "MAX_PROMPT_LEN": 1024,
        "MIN_RESPONSE_LEN": 128,      # Critical: Forces static output buffer
        "PREFILL_HINT": "STATIC",     # Critical: Prevents dynamic input shapes
        "GENERATE_HINT": "BEST_PERF",
        "PERFORMANCE_HINT": "LATENCY",
        "CACHE_DIR": cache_dir
    }

    try:
        pipe = ov_genai.LLMPipeline(model_path, "NPU", **config)
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # 2. EXECUTION LOOP
    print(f"Success! Running tests. Results will be saved to {report_file}")
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# NPU Performance & Accuracy Report\n\n")
        f.write("| Question | Tokens/Sec | Response |\n")
        f.write("| :--- | :--- | :--- |\n")

        for i, q in enumerate(test_questions):
            print(f"[{i+1}/{len(test_questions)}] Processing: {q}")
            
            start_time = time.time()
            full_response = ""
            first_token_time = None
            token_count = 0

            def streamer(subword):
                nonlocal first_token_time, token_count, full_response
                if first_token_time is None:
                    first_token_time = time.time()
                full_response += subword
                token_count += 1
                return ov_genai.StreamingStatus.RUNNING

            try:
                # do_sample=False is often more stable for benchmark runs
                pipe.generate(format_prompt(q), max_new_tokens=128, streamer=streamer, do_sample=False)
                
                end_time = time.time()
                # Ensure we don't divide by zero if first_token_time is None
                gen_duration = end_time - first_token_time if first_token_time else 0.001
                tps = token_count / gen_duration

                # Clean response for Markdown table format
                clean_res = full_response.replace("\n", " ").replace("|", "/").strip()
                f.write(f"| {q} | {tps:.2f} | {clean_res} |\n")
            except Exception as e:
                f.write(f"| {q} | ERROR | {str(e)} |\n")

    print(f"\nBenchmark Complete! Open {report_file} to see the results.")

if __name__ == "__main__":
    run_automated_test()
