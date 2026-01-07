import string
import torch

# --- Sector Definitions ---
DIGITS = string.digits
# Use your original definition for CHARS to ensure the order matches
CHARS = string.digits + string.ascii_lowercase + string.ascii_uppercase

# --- Captcha Dimensions (RESTORED TO YOUR ORIGINALS) ---
WIDTH = 200
HEIGHT = 80
CAPTCHA_LENGTH = 6

# --- Training Constants ---
BATCH_SIZE = 16
DEVICE = "xpu"

# This ensures the model output layer matches our character set (62)
NUM_CLASSES = len(CHARS)
