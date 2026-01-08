import torch
import string

# --- Hardware Configuration ---
DEVICE = "xpu" if torch.xpu.is_available() else "cpu"

# --- Image Specifications ---
WIDTH = 200
HEIGHT = 80
CAPTCHA_LENGTH = 6

# --- Character Sets ---
DIGITS = string.digits
CHARS = string.digits + string.ascii_lowercase + string.ascii_uppercase

# --- Training Hyperparameters ---
BATCH_SIZE = 16 
LEARNING_RATE = 0.001
