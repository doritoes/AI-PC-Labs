import os

# NPU Environment Flags
os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"

# Model Path - Update if your path is different
MODEL_PATH = os.path.join(os.environ['USERPROFILE'], 'Edge-AI', 'models', 'qwen-1.5b')

# NPU Pipeline Configuration
NPU_CONFIG = {
    "MAX_PROMPT_LEN": 1024,
    "MIN_RESPONSE_LEN": 10,
    "PREFILL_HINT": "STATIC",
    "GENERATE_HINT": "BEST_PERF",
    "PERFORMANCE_HINT": "LATENCY",
    "CACHE_DIR": "npu_cache"
}
