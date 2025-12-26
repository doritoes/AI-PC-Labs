# Local Chatbot
We are going to run a local chatbot. Large Language Models (LLMs) are typically "memory bound". The NPU's efficiency allows it to stream text while using less power than the CPU.

Phi-3-Mini's strengths lie in its impressive performance for its small size, offering strong reasoning, coding, and math abilities, making it highly capable for on-device or resource-constrained applications, excellent instruction following, and efficient, cost-effective operation, often outperforming larger models across benchmarks. It excels at logic, understanding complex instructions, and handling long contexts (with the 128k version), making it powerful for tasks like RAG and code generation without needing massive hardware. 

## Quantization
A "full-weight" LLM is like a high-resolution 4K movie, while a Quantized (INT4) model is like a compressed 720p version. It is smaller (2GB vs 8GB+) and faster, but keeps enough "intelligence" to chat fluently.

## Downloading a Quantized Phi-3 Model
Phi-3-mini is a very capable and small LLM. It is possible to download the full model and quantize it yourself:
- `pip install optimum-intel[openvino,nncf]`
- `pip install huggingface_hub[hf_xet]`
- `optimum-cli export openvino --model microsoft/Phi-3-mini-4k-instruct --task text-generation-with-past --weight-format int4 --group-size 128 --ratio 1.0 "$env:USERPROFILE\Edge-AI\models\phi-3-mini-int4"`
- WARNING there seems to be a conflict with how optimum-cli handles the "stateful" parameters of Phi-3

To avoid issues, we will download the pre-quantized INT4 version
- Activate the environment
  - `cd $env:USERPROFILE\Edge-AI`
  - `.\nputest_env\Scripts\Activate.ps1`
- `hf download OpenVINO/Phi-3-mini-4k-instruct-int4-ov --local-dir "$env:USERPROFILE\Edge-AI\models\phi-3-mini-int4"`
- Weight compression: original model FP16 (16 bits per number), converting to INT4 (4 bits per number)
- Memory footprint: shrings from ~7GB to ~2.2GB
- Without shrinking, the model wouldn't fit into the NPU's dedicated high-speed cache, forcing it to use slower system RAM

## Create the NPU Chatbot Script

chatbot_npu.py
~~~
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
    pipe = ov_genai.LLMPipeline(model_path, "NPU", **config)

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
~~~

## Running the Chatbot
1. Enable the "emergency bypass" to use standard system memory when the driver can't allocate the specialsed "Level Zero" (L0) memory
    - `$env:DISABLE_OPENVINO_GENAI_NPU_L0=1`
2. `python chatbot_npu.py`
    - note the 20-40 second delay the first time it is run. The NPU is "compiling" the model, turing generic AI code into specialized instructions for the processor
3. Enter a series of prompts and record the metrics
    - TTFT (time to first token) How long it takes for the AI to "start" talking
    - TPS (tokens per second) The reading speed. Human reading speed is ~5-8 tokens/second
  
    - 
Goal: Run a Large Language Model (LLM) like Phi-3 or Llama-3 entirely offline.

Process: Use the openvino-genai library (which you installed) to load a quantized (INT4) model.

Measurement: "Tokens per second" and "Time to First Token" (TTFT).

Key Learning: Understanding Quantization. Students learn how to shrink a multi-gigabyte model into a 2GB "weight" file so it fits into the NPU's memory. They will use the BEST_PERF hint to optimize the NPU's power.

Script Snippet: Python

import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline("model_path", "NPU")
print(pipe.generate("Explain Edge AI in three sentences.", max_new_tokens=100))


