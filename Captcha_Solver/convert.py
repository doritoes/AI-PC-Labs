import torch
import openvino as ov
from train import SimpleCNN

# 1. Load the "High Accuracy" weights
model = SimpleCNN()
model.load_state_dict(torch.load("captcha_model.pth"))
model.eval()

# 2. Convert to OpenVINO IR Format
# We use FP16 as a middle step before the NPU's INT8 quantization
print("Converting PyTorch model to OpenVINO IR...")
example_input = torch.randn(1, 1, 60, 160)
ov_model = ov.convert_model(model, example_input=example_input)

# 3. Save the model files
ov.save_model(ov_model, "captcha_model.xml")
print("SUCCESS: captcha_model.xml and captcha_model.bin created.")
