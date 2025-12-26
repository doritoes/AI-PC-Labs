import openvino as ov
import numpy as np
import time
import os
from transformers import AutoTokenizer

def run_manual_npu():
    # 1. ENVIRONMENT & PATHS
    os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"
    model_dir = os.path.join(os.environ['USERPROFILE'], 'Edge-AI', 'models', 'tiny-llama')
    model_xml = os.path.join(model_dir, "openvino_model.xml")
    
    print("\n--- STARTING BARE-METAL NPU INITIALIZATION ---")
    
    # 2. LOAD TOKENIZER (Uses CPU)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # 3. INITIALIZE OPENVINO CORE
    core = ov.Core()
    
    print("Step 1: Reading Model IR...")
    model = core.read_model(model_xml)

    print("Step 2: Compiling Model for NPU (This is the critical step)...")
    try:
        # We use a very basic config to avoid driver triggers
        compiled_model = core.compile_model(model, "NPU")
        infer_request = compiled_model.create_infer_request()
    except Exception as e:
        print(f"\n[HARDWARE ERROR]: The NPU driver failed to compile the model: {e}")
        print("Switching to 'AUTO' or 'CPU' is the only remaining path for this driver version.")
        return

    # 4. CHAT INTERFACE
    print("\n" + "="*50)
    print("NPU MANUAL CHAT ONLINE")
    print("="*50)

    while True:
        user_input = input("\nStudent: ")
        if user_input.lower() in ['quit', 'exit']: break

        # Simple prompt formatting for TinyLlama
        prompt = f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{user_input}</s>\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="np")
        input_ids = inputs['input_ids']

        print("NPU Assistant: ", end="", flush=True)
        
        # Generation loop (Manual Argmax)
        generated_ids = input_ids.tolist()[0]
        for _ in range(100):
            # Run inference on NPU
            results = compiled_model(input_ids)[0]
            next_token_id = np.argmax(results[:, -1, :])
            
            if next_token_id == tokenizer.eos_token_id:
                break
                
            word = tokenizer.decode([next_token_id])
            print(word, end="", flush=True)
            
            generated_ids.append(next_token_id)
            input_ids = np.array([generated_ids])

        print("\n" + "-"*40)

if __name__ == "__main__":
    run_manual_npu()
