import openvino_genai as ov_genai
import time
import os

def run_npu_chatbot():
    # 1. HARDWARE ENVIRONMENT STABILITY
    os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"
    
    model_path = os.path.join(os.environ['USERPROFILE'], 'Edge-AI', 'models', 'qwen-1.5b')

    # 2. INITIALIZE NPU (Static Shape Enforcement)
    print("\n" + "-"*50)
    print("INITIALIZING NPU: FORCING STATIC SHAPE MODE")
    print("-" * 50)
    
    # NPU_USE_NPUW: NO - prevents the MatMul map error
    # NPU_COMPILATION_MODE_HINT: static - fixes the 'dynamic shape' crash
    # PERFORMANCE_HINT: LATENCY - ensures speed
    config = {
        "NPU_USE_NPUW": "NO",
        "NPU_COMPILATION_MODE_HINT": "static",
        "PERFORMANCE_HINT": "LATENCY"
    }

    try:
        # We load the model with the static hint
        pipe = ov_genai.LLMPipeline(model_path, "NPU", **config)
    except Exception as e:
        print(f"\n[CRITICAL ERROR]: {e}")
        print("\nTEACHER NOTE: If this fails, the Qwen model export itself is incompatible with NPU static requirements.")
        return

    # 3. CHAT INTERFACE
    print("\n" + "="*50)
    print("NPU CHATBOT ONLINE (Static Mode)")
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

        try:
            # We limit the max tokens to 128 to ensure it stays in the static buffer
            pipe.generate(prompt, max_new_tokens=128, streamer=streamer)
        except Exception as e:
            print(f"\n[GENERATION ERROR]: {e}")
            break

        print(f"\n\n(NPU Stats: {token_count} tokens generated)")

if __name__ == "__main__":
    run_npu_chatbot()
