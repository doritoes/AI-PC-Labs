import os
import torch
import nncf
import openvino as ov
from torch.utils.data import DataLoader
from train import CaptchaModel, CaptchaDataset
from config import WIDTH, HEIGHT, DATASET_SIZE

def quantize():
    # 1. Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    model_path = os.path.join(root_dir, "captcha_model.pth")
    int8_model_dir = os.path.join(root_dir, "openvino_int8_model")
    
    if not os.path.exists(int8_model_dir):
        os.makedirs(int8_model_dir)

    # 2. Load the PyTorch Model
    print("🔄 Loading PyTorch model for quantization...")
    model = CaptchaModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 3. Prepare Calibration Dataset
    # Quantization requires a small subset of data to "calibrate" the 8-bit ranges
    print("📊 Preparing calibration data (Post-Training Quantization)...")
    
    # We use the CaptchaDataset class we already built
    dataset_path = os.path.join(root_dir, "dataset")
    if not os.path.exists(dataset_path):
        print("❌ Error: Dataset folder not found. Please run training first.")
        return

    calibration_dataset = CaptchaDataset(dataset_path)
    # We only need about 100-300 samples for high-quality calibration
    calibration_loader = DataLoader(calibration_dataset, batch_size=1, shuffle=True)

    def transform_fn(data_item):
        images, _ = data_item
        return images.numpy()

    # Wrap the loader for NNCF
    calibration_data = nncf.Dataset(calibration_loader, transform_fn)

    # 4. Run NNCF Quantization
    print("🚀 Running NNCF INT8 Quantization (this may take a minute)...")
    quantized_model = nncf.quantize(model, calibration_data)

    # 5. Convert to OpenVINO IR and Save
    print("💾 Saving INT8 model...")
    dummy_input = torch.randn(1, 1, HEIGHT, WIDTH)
    ov_model = ov.convert_model(quantized_model, example_input=dummy_input)
    
    ov_xml_path = os.path.join(int8_model_dir, "captcha_model_int8.xml")
    ov.save_model(ov_model, ov_xml_path)

    print("-" * 50)
    print(f"✅ INT8 Quantization Successful!")
    print(f"📁 Folder: {int8_model_dir}")
    print(f"💡 This model is now optimized for the Arrow Lake NPU's INT8 pipelines.")
    print("-" * 50)

if __name__ == "__main__":
    # Ensure nncf is installed: pip install nncf
    quantize()
