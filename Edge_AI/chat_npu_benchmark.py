import os
import time
import openvino_genai as ov_genai

def format_prompt(user_query):
    system_message = "You are a precise AI Assistant. Provide concise, logical answers."
    return f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"

def run_automated_test():
    # --- CRITICAL HARDWARE FLAGS ---
    # This environment variable is mandatory for Intel NPU driver stability
    os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"

    model_path = os.path.join(os.environ['USERPROFILE'], 'Edge-AI', 'models', 'qwen-1.5b')
    cache_dir = os.path.join(os.getcwd(), 'npu_cache')
    report_file = "npu_test_results.md"

    # Test suite for Logic, Science, and Creativity
    test_questions = [
        "What is the square root of 144?",
        "If all bloops are blips and all blips are blops, are all bloops blops?",
        "Explain photosynthesis in one sentence.",
        "Write a 4-line poem about a computer chip.",
        "How many legs does a spider have?"
    ]

    print("\n" + "="*50)
    print("  NPU AUTOMATED BENCHMARK (Qwen 2.5)  ")
    print("="*50)

    # Configuration proven to work on your specific v4515 driver
    pipeline_config = {
        "MAX_PROMPT_LEN": 1024,
        "MIN_RESPONSE_LEN": 128,
        "PREFILL_HINT": "STATIC",
        "GENERATE_HINT": "BEST_PERF",
        "PERFORMANCE_HINT": "LATENCY",
        "CACHE_DIR": cache_dir
    }

    try:
        # Initializing the pipeline
        pipe = ov_genai.LLMPipeline(model_path, "NPU", **pipeline_config)
    except Exception as e:
        print(f"\n[CRITICAL ERROR]: {e}")
        return

    print(f"\nInitialization Success! Processing {len(test_questions)} tests...")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# NPU Performance & Accuracy Report\n\n")
        f.write("| Question | TTFT (ms) | Speed (TPS) | Response |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")

        for i, q in enumerate(test_questions):
            print(f"[{i+1}/{len(test_questions)}] Querying: {q}")

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
                pipe.generate(format_prompt(q), max_new_tokens=256, streamer=streamer, do_sample=False)

                end_time = time.time()

                # Metrics Calculation
                ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
                gen_duration = end_time - first_token_time if first_token_time else 0.001
                tps = token_count / gen_duration

                # Clean text for Markdown compatibility
                clean_res = full_response.replace("\n", " ").replace("|", "/").strip()
                f.write(f"| {q} | {ttft_ms:.1f} | {tps:.1f} | {clean_res} |\n")

            except Exception as e:
                f.write(f"| {q} | ERROR | ERROR | {str(e)} |\n")

    print("\n" + "*"*50)
    print(f" TEST COMPLETE: Results saved to {report_file}")
    print("*"*50)

if __name__ == "__main__":
    run_automated_test()
