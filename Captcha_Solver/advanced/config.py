# advanced/config.py
import string

CHARS = string.digits + string.ascii_lowercase + string.ascii_uppercase
CAPTCHA_LENGTH = 6
WIDTH, HEIGHT = 200, 80  # Increased size for 6 characters
DATASET_SIZE = 20000      # Base images, add 1:2 augmentation for total 60k
TOTAL_SIZE = 60000      # With 1:2 augmentation (Rotate/Blur)
BATCH_SIZE = 32 # reduced to work in reduced RAM
DEVICE = "xpu"           # Intel iGPU/dGPU via IPEX
