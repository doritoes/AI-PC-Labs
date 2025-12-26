# Local Chatbot
The Local Chatbot (OpenVINO GenAI)

Goal: Run a Large Language Model (LLM) like Phi-3 or Llama-3 entirely offline.

Process: Use the openvino-genai library (which you installed) to load a quantized (INT4) model.

Measurement: "Tokens per second" and "Time to First Token" (TTFT).

Key Learning: Understanding Quantization. Students learn how to shrink a multi-gigabyte model into a 2GB "weight" file so it fits into the NPU's memory. They will use the BEST_PERF hint to optimize the NPU's power.

Script Snippet: Python

import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline("model_path", "NPU")
print(pipe.generate("Explain Edge AI in three sentences.", max_new_tokens=100))
