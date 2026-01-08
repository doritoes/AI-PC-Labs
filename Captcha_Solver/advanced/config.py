import torch
import string

# --- Hardware Configuration ---
DEVICE = "xpu" if torch.xpu.is_available() else "cpu"

# --- Image Specifications ---
WIDTH = 200
HEIGHT = 80
CAPTCHA_LENGTH = 6

# --- Character Sets (The Model's Vocabulary) ---
DIGITS = string.digits
# 0-9 (10) + a-z (26) + A-Z (26) = 62 total classes
CHARS = string.digits + string.ascii_lowercase + string.ascii_uppercase

# --- Training Hyperparameters ---
BATCH_SIZE = 16 
LEARNING_RATE = 0.001
