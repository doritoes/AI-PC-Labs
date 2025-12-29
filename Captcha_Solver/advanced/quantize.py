import os
import torch
import nncf
import openvino as ov
from torch.utils.data import DataLoader, Subset
from train import CaptchaModel, CaptchaDataset
from config import WIDTH, HEIGHT

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

    # 3. Prepare Calibration Dataset (Limited to 300 samples for speed)
    dataset_path = os.path.join(root_dir, "dataset")
    if not os.path.exists(dataset_path):
        print("❌ Error: Dataset folder not found. Please run training first.")
        return

    full_dataset = CaptchaDataset(dataset_path)
    # Subset to 300 random images for calibration
    indices = torch.randperm(len(full_dataset))[:300]
    calibration_subset = Subset(full_dataset, indices)
    calibration_loader = DataLoader(calibration_subset, batch_size=1, shuffle=False)

    def transform_fn(data_item):
        images, _ = data_item
        return images.numpy()

    # Wrap for NNCF
    calibration_data = nncf.Dataset(calibration_loader, transform_fn)

    # 4. Run NNCF Quantization
    print("🚀 Running NNCF INT8 Quantization (Optimizing for NPU Tiles)...")
    # This creates a quantized version of the PyTorch model
    quantized_model = nncf.quantize(model, calibration_data)

    # 5. Convert to Static OpenVINO IR
    print("💾 Saving Static INT8 model...")
    # Explicitly defining the static input shape
    static_input = torch.randn(1, 1, HEIGHT, WIDTH)
    
    # example_input freezes the model to these exact dimensions
    ov_model = ov.convert_model(quantized_model, example_input=static_input)
    
    ov_xml_path = os.path.join(int8_model_dir, "captcha_model_int8.xml")
    ov.save_model(ov_model, ov_xml_path)

    print("-" * 50)
    print(f"✅ Static INT8 Quantization Successful!")
    print(f"📁 Model Location: {int8_model_dir}")
    print(f"🚀 Best Hardware: Intel AI Boost (NPU)")
    print("-" * 50)

if __name__ == "__main__":
    quantize()
