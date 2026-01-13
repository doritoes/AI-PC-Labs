"""
quantize model to INT8 for NPU use
"""
import openvino as ov
import nncf
from torch.utils.data import DataLoader
from train import CaptchaDataset  # Import your dataset class

# 1. Load the FP32 OpenVINO model
print("Loading FP32 Model...")
core = ov.Core()
model = core.read_model("captcha_model.xml")

# 2. Prepare Calibration Data
# We only need a small sample (100-300 images) to "calibrate" the weights
calibration_dataset = CaptchaDataset(size=300)
calibration_loader = DataLoader(calibration_dataset, batch_size=1)

# Transform the loader into the format NNCF expects
def transform_fn(data_item):
    """ return numpy object """
    images, _ = data_item
    return images.numpy()

calibration_data = nncf.Dataset(calibration_loader, transform_fn)

# 3. Quantize the model
print("Starting INT8 Quantization... (Calibrating weights)")
quantized_model = nncf.quantize(model, calibration_data)

# 4. Save the new INT8 model
ov.save_model(quantized_model, "captcha_model_int8.xml")
print("-" * 40)
print("SUCCESS: INT8 Quantized model saved!")
print("Files: captcha_model_int8.xml and captcha_model_int8.bin")
print("-" * 40)
