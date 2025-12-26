import openvino_genai as ov_genai
import time
import os

def run_npu_chatbot():
    os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"
    model_path = os.path.join(os.environ['USERPROFILE'], 'Edge-AI', 'models', 'qwen-1.5b')

    print("\n" + "-"*50)
    print("INITIALIZING NPU: 2025.x STABILITY MODE")
    print("-" * 50)
    
    # 2025.x Specific Config:
    # PREFILL_HINT: STATIC forces the NPU to use a fixed buffer for the initial prompt
    # GENERATE_HINT: BEST_PERF ensures the hardware stays in high-power state
    config = {
        "NPU_USE_NPUW": "NO",
        "PREFILL_HINT": "STATIC",
        "GENERATE_HINT": "BEST_PERF",
        "PERFORMANCE_HINT": "LATENCY"
    }

    try:
        # Load the pipeline with the new 2025 property keys
        pipe = ov_genai.LLMPipeline(model_path, "NPU", **config)
    except Exception as e:
        print(f"\n[CRITICAL ERROR]: {e}")
        return

    print("\n" + "="*50)
    print("NPU CHATBOT ONLINE (Qwen 2.5)")
    print("="*50)

    while True:
        prompt = input("\nStudent: ")
        if prompt.lower() in ['quit', 'exit']: break

        print("NPU Assistant: ", end="", flush=True)

        def streamer(subword):
            print(subword, end="", flush=True)
            return ov_genai.StreamingStatus.RUNNING

        try:
            # We keep max_new_tokens relatively small to fit the static buffer
            pipe.generate(prompt, max_new_tokens=128, streamer=streamer)
        except Exception as e:
            print(f"\n[GENERATION ERROR]: {e}")
            break
        print("\n")

if __name__ == "__main__":
    run_npu_chatbot()
