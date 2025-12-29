import os
import sys
import torch
import openvino as ov
from train import SimpleCNN

# 1. Load the "High Accuracy" weights
model = SimpleCNN()
if os.path.exists("captcha_model.pth"):
    model.load_state_dict(torch.load("captcha_model.pth"))
    model.eval()
    print("Loaded weights from captcha_model.pth")
else:
    print("Error: captcha_model.pth not found! Run train.py first.")
    sys.exit()

# 2. ADD THIS: Define the exact shape [Batch, Channels, Height, Width]
input_shape = [1, 1, 60, 160]

# 3. Convert to OpenVINO IR Format
print("Converting PyTorch model to OpenVINO IRi with fixed input shape for NPU compatibility...")
ov_model = ov.convert_model(model, example_input=torch.randn(1, 1, 60, 160))
# Set the shape explicitly in the model metadata
ov_model.reshape({0: input_shape})

# 4. Save the model files
ov.save_model(ov_model, "captcha_model.xml")
print("-" * 40)
print("SUCCESS: Model converted to OpenVINO IR!")
print("Files created: captcha_model.xml and captcha_model.bin")
print("-" * 40)
