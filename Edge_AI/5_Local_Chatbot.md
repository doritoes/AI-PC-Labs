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
[chatbot_npu.py](chatbot_npu.py)

## Running the Chatbot
1. Enable the "emergency bypass" to use standard system memory when the driver can't allocate the specialized "Level Zero" (L0) memory
    - `$env:DISABLE_OPENVINO_GENAI_NPU_L0=1`
2. `python .\chatbot_npu.py`
    - That "long time" you're experiencing is actually a healthy sign that the hardware is working—this phase is known as First Ever Inference Latency (FEIL). Because the NPU is a specialized "streaming" processor rather than a general-purpose one like your CPU, it has to physically map the logic of the Phi-3 model into its own hardware-level execution circuits. For an LLM with billions of parameters, this often takes 2 to 5 minutes on the first run.
    - The NPU is "compiling" the model, turning generic AI code into specialized instructions for the processor
    - Task Manager > Performance; note the CPU is working, and later the massive write to disk
    - The NPU is "building" the model's math structure for the first time. Once this is done, it is cached, and future starts will be nearly instant.
4. Enter a series of prompts and record the metrics
    - TTFT (time to first token) How long it takes for the AI to "start" talking
    - TPS (tokens per second) The reading speed. Human reading speed is ~5-8 tokens/second

## Example in Python Script

~~~
import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline("model_path", "NPU")
print(pipe.generate("Explain Edge AI in three sentences.", max_new_tokens=100))
~~~

