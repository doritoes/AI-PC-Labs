import openvino_genai as ov_genai
import time
import os

def run_npu_chatbot():
    # Force NPU to ignore newer memory allocation methods that cause driver mismatch
    os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"
    
    model_path = os.path.join(os.environ['USERPROFILE'], 'Edge-AI', 'models', 'qwen-1.5b')

    print("\n" + "-"*50)
    print("INITIALIZING NPU: FORCING STATIC CONTAINERS")
    print("-" * 50)
    
    # These parameters are the 'Silver Bullet' for the dynamic shape error.
    # We define exactly how much 'space' the NPU should reserve.
    pipeline_config = {
        "MAX_PROMPT_LEN": 1024,      # Fixed input buffer size
        "MIN_RESPONSE_LEN": 128,     # Fixed output buffer size
        "PREFILL_HINT": "STATIC",    # Disables dynamic prompt execution
        "GENERATE_HINT": "BEST_PERF" # Optimizes for NPU throughput
    }

    try:
        # Load the pipeline with the specific static container instructions
        pipe = ov_genai.LLMPipeline(model_path, "NPU", pipeline_config)
    except Exception as e:
        print(f"\n[CRITICAL ERROR]: {e}")
        print("\nNOTE: If this still fails, your model export lacks the 'Stateful' metadata required for NPU.")
        return

    print("\n" + "="*50)
    print("NPU CHATBOT ONLINE")
    print("="*50)

    while True:
        prompt = input("\nStudent: ")
        if prompt.lower() in ['quit', 'exit']: break

        print("NPU Assistant: ", end="", flush=True)

        def streamer(subword):
            print(subword, end="", flush=True)
            return ov_genai.StreamingStatus.RUNNING

        try:
            # do_sample=False is often required for NPU stability in early drivers
            pipe.generate(prompt, max_new_tokens=256, streamer=streamer, do_sample=False)
        except Exception as e:
            print(f"\n[GENERATION ERROR]: {e}")
            break
        print("\n")

if __name__ == "__main__":
    run_npu_chatbot()
