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

# 2. Convert to OpenVINO IR Format
print("Converting PyTorch model to OpenVINO IR... (This is fast)")
# Input shape: [Batch, Channel, Height, Width]
example_input = torch.randn(1, 1, 60, 160)
ov_model = ov.convert_model(model, example_input=example_input)

# 3. Save the model files
ov.save_model(ov_model, "captcha_model.xml")
print("-" * 40)
print("SUCCESS: Model converted to OpenVINO IR!")
print("Files created: captcha_model.xml and captcha_model.bin")
print("-" * 40)
