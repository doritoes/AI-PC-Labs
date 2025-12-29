# advanced/config.py
import string

CHARS = string.digits + string.ascii_lowercase + string.ascii_uppercase
CAPTCHA_LENGTH = 6
WIDTH, HEIGHT = 200, 80  # Increased size for 6 characters
DATASET_SIZE = 50000      # Base images
TOTAL_SIZE = 100000      # With 1:1 augmentation (Rotate/Blur)
BATCH_SIZE = 128
DEVICE = "xpu"           # Intel iGPU/dGPU via IPEX
