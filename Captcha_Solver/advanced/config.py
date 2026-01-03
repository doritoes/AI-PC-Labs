import string

# --- Dataset Settings ---
# 20k is the "Goldilocks" zone for 16GB RAM: enough variety, fits in memory.
DATASET_SIZE = 20000 
CAPTCHA_LENGTH = 6
CHARS = string.digits + string.ascii_lowercase + string.ascii_uppercase
WIDTH = 200
HEIGHT = 80

# --- Training Hyperparameters ---
# Small batches (32) provide "noisy" gradients which help escape mode collapse.
# It also keeps the XPU memory pressure low.
BATCH_SIZE = 32

# Aggressive Learning Rate to kick the model off the 4.12 floor.
# We use 0.001 (1e-3) as a stable breakout start point.
LEARNING_RATE = 0.001

# 50 Epochs is the target for a meaningful descent curve.
EPOCHS = 50

# --- Hardware Settings ---
DEVICE = "xpu"  # Targeted for Intel Xe2 iGPU
