# Local Chatbot
We are going to run a local chatbot. Large Language Models (LLMs) are typically "memory bound". The NPU's efficiency allows it to stream text while using less power than the CPU.

Phi-3-Mini's strengths lie in its impressive performance for its small size, offering strong reasoning, coding, and math abilities, making it highly capable for on-device or resource-constrained applications, excellent instruction following, and efficient, cost-effective operation, often outperforming larger models across benchmarks. It excels at logic, understanding complex instructions, and handling long contexts (with the 128k version), making it powerful for tasks like RAG and code generation without needing massive hardware. 

## Quantization
A "full-weight" LLM is like a high-resolution 4K movie, while a Quantized (INT4) model is like a compressed 720p version. It is smaller (2GB vs 8GB+) and faster, but keeps enough "intelligence" to chat fluently.

## Updating to the Latest Driver
1. Open device manager
2. Expand Neural processors
3. Double-click on Intel(R) AI Boost
4. Note the driver date and version
    - `2/6/2025` Driver Version 32.0.100.3717
5. Download the installer for the new driver from
    - https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html
    - In my lab I selected v4515 12/19/2025
6. Install the driver
    - Run the .exe
    - Reboot
7. Always clean cache after a driver update
    - `Remove-Item -Recurse -Force "$env:LOCALAPPDATA\OpenVINO\cache"`

## Downloading a Pre-Quantized Qwen2.5-1.5B Model
Qwen2.5-1.5B is a dense, causal language model and part of the latest generation of the Qwen series developed by Alibaba Cloud. It is specifically engineered to be a "small-language model" (SLM) that delivers high-performance reasoning on edge devices like your Intel NPU.
- Instruction Following: It adheres to "system prompts" much better than older 1B models
- Coding: It can generate and debug Python snippets locally on your NPU
- Role play/Tone: It can shift its persona (e.g., "Answer like a scientist") with high consistency

- Activate the environment
  - `cd $env:USERPROFILE\Edge-AI`
  - `.\nputest_env\Scripts\Activate.ps1`
- `hf download OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov --local-dir "$env:USERPROFILE\Edge-AI\models\qwen-1.5b"`
  - Weight compression: original model FP16 (16 bits per number), converting to INT4 (4 bits per number)
  - Memory footprint: shrinks from ~3.1GB to ~950MB
  - Without shrinking, the model wouldn't fit into the NPU's dedicated high-speed cache, forcing it to use slower system RAM

## Create the NPU Chatbot Script
[chatbot_npu.py](chatbot_npu.py)

## Running the Chatbot
1. `python .\chatbot_npu.py`
    - That "long time" you're experiencing is actually a healthy sign that the hardware is working—this phase is known as First Ever Inference Latency (FEIL). Because the NPU is a specialized "streaming" processor rather than a general-purpose one like your CPU, it has to physically map the logic of the Phi-3 model into its own hardware-level execution circuits. For an LLM with billions of parameters, this often takes 2 to 5 minutes on the first run.
    - The NPU is "compiling" the model, turning generic AI code into specialized instructions for the processor
    - Task Manager > Performance; note the CPU is working, and later the massive write to disk
    - The NPU is "building" the model's math structure for the first time. Once this is done, it is cached, and future starts will be nearly instant.
2. Enter a series of prompts and record the metrics
    - TTFT (time to first token) How long it takes for the AI to "start" talking
    - TPS (tokens per second) The reading speed. Human reading speed is ~5-8 tokens/second

NOTE If you get a crash, make sure you have the python environment activated. If you need to, you can clear the cache and make it recompile for the NPU: `Remove-Item -Recurse -Force "$env:LOCALAPPDATA\OpenVINO\cache" -ErrorAction SilentlyContinue`

Some test prompts:
1. Reading & Math
    - If I have 3 oranges and you give me 2 more, but then I eat one and give half of what's left to my friend, how many oranges do I have? Walk me through your thinking step-by-step.
2. Creativity and Context
    - Write a short sci-fi story about a world where everyone has a personal NPU chip in their brain that helps them remember everything perfectly, but one day yours starts to glitch.
3. Concise Summarization
    - Explain the difference between a CPU and an NPU to a 5-year-old in exactly three bullet points.
4. Math and LLM Mistakes
    - How can I convert 12 degrees Fahrenheit to Celsius?
    - NOTE how the LLM gets it incorrect! (should be -11.1111 degrees C)
        - Tokenization vs. Math: LLMs don't "do math" with a calculator; they predict the next most likely piece of text (token). Because $32 \times \frac{5}{9}$ is a very common snippet in training data, the model's "attention" was pulled toward calculating that first, violating the parentheses.
        - Quantization Impact: Using INT4 (4-bit) makes the model fast and small, but it can slightly degrade its ability to follow complex logical constraints compared to the full-sized FP16 version.
5. Stump the Model
    - I mixed 1 cup of brazil nuts, 1 cup of raw almonds, and 1 cup of whole walnuts together in a large container. I then removed 1 cup of nuts from the large container. What is the contents of the cup of nuts i took out?

## Give the Chatbot a Goal
To make this a true "AI Assistant" lab, you can wrap the user's input in a template that gives the AI a personality or a specific goal. This is how "ChatGPT" is tuned to be helpful rather than just completing text.

Here we add the "format_prompt" function to tell the model how to behave before it sees the user question.
- [chatbot_npu_goal.py](chatbot_npu_goal.py)
- `python .\chatbot_npu_goal.py`

## Example in Python Script
Use this is an example of querying the model from a python script.
[chat_npu_benchmark.py](chat_npu_benchmark.py)

## Final Learnings Review
1. Dedicated Hardware: You aren't using the CPU (the "Generalist") or the GPU (the "Artist"). You are using the NPU (the "Specialist") specifically designed for the matrix math that AI requires.
2. Quantization: By turning complex numbers into simple 4-bit integers, we made a 1.5-billion-parameter brain small enough to fit in a laptop's pocket.
3. Local Privacy: Notice that your Wi-Fi could be turned off right now, and this would still work. No data left the room.
4. Static Memory: We learned that NPUs prefer a "reserved seat" (Static Containers) rather than "finding a seat" (Dynamic Shapes) like a CPU does.
