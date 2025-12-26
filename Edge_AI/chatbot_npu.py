import openvino_genai as ov_genai
import time
import os
import shutil

def run_npu_chatbot():
    # 1. HARDWARE ENVIRONMENT FIXES
    # Force the L0 memory bypass to prevent 'Out of Host Memory' errors
    os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"
    
    # 2. PATH SETUP
    model_path = os.path.join(os.environ['USERPROFILE'], 'Edge-AI', 'models', 'phi-3-mini-int4')
    cache_path = os.path.join(os.environ['LOCALAPPDATA'], 'OpenVINO', 'cache')

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return

    # 3. INITIALIZE NPU WITH MODERN CONFIGURATION
    print("\n--- Initializing Intel NPU (AI PC Accelerator) ---")
    
    # NOTE: We removed 'NPUW_CONF' and replaced it with direct NPU properties
    # to match the latest 2025 Intel NPU drivers.
    config = {
        "NPU_RUN_INFERENCES_SEQUENTIALLY": "OFF",
        "PERFORMANCE_HINT": "LATENCY"
    }

    try:
        # Loading the pipeline directly to the NPU
        pipe = ov_genai.LLMPipeline(model_path, "NPU", **config)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] NPU Initialization failed: {e}")
        if "NPUW_CONF" in str(e):
            print("Tip: Your driver requires direct NPU properties instead of NPUW_CONF.")
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
            return ov_genai.StreamingStatus.RUNNING

        # 6. GENERATION
        try:
            pipe.generate(prompt, max_new_tokens=256, streamer=streamer)
        except Exception as e:
            print(f"\n[GENERATION ERROR]: {e}")
            break

        # 7. METRICS
        end_time = time.time()
        ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
        gen_duration = end_time - first_token_time if first_token_time else 0
        tps = token_count / gen_duration if gen_duration > 0 else 0

        print(f"\n\n" + "-"*40)
        print(f"NPU PERFORMANCE METRICS:")
        print(f">> Time to First Token (TTFT): {ttft:.2f} ms")
        print(f">> Tokens Per Second (TPS):    {tps:.2f} tok/s")
        print(f"-"*40)

if __name__ == "__main__":
    run_npu_chatbot()
