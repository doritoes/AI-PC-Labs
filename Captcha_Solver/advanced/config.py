import string

# --- Dataset Settings ---
DATASET_SIZE = 6000 
CAPTCHA_LENGTH = 6
# Keep the full set here so the model architecture creates all 62 output nodes
CHARS = string.digits + string.ascii_lowercase + string.ascii_uppercase
WIDTH = 200
HEIGHT = 80

# --- Training Hyperparameters ---
BATCH_SIZE = 16
# Increased slightly to help the model "jump" during the Digits phase
LEARNING_RATE = 0.002 
# 50 Epochs total (5 Digits + 45 Full Alphanumeric)
EPOCHS = 50

# --- Hardware Settings ---
DEVICE = "xpu"
