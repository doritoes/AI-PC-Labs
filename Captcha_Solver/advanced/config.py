import string

# --- Dataset Settings ---
# We are moving to 40k unique images with 0 augmentations
DATASET_SIZE = 40000 
CAPTCHA_LENGTH = 6
CHARS = string.digits + string.ascii_lowercase + string.ascii_uppercase
WIDTH = 200
HEIGHT = 80

# --- Training Hyperparameters ---
# Dropping Batch Size to 16 to prevent the "16,000s" disk-swapping spikes
BATCH_SIZE = 16 
# Increased Learning Rate to force the model off the 4.12 loss floor
LEARNING_RATE = 0.004 
EPOCHS = 35

# --- Hardware Settings ---
DEVICE = "xpu"  # Targeted for Intel Xe2 iGPU
