import os
import cv2
import numpy as np
import openvino as ov
from config import CHARS, CAPTCHA_LENGTH, WIDTH, HEIGHT

def solve_captcha(image_path, model_type="int8"):
    # 1. Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    if model_type == "int8":
        model_xml = os.path.join(root_dir, "openvino_int8_model", "captcha_model_int8.xml")
    else:
        model_xml = os.path.join(root_dir, "openvino_model", "captcha_model.xml")

    if not os.path.exists(model_xml):
        print(f"❌ Model not found at {model_xml}. Run convert.py/quantize.py first.")
        return None

    # 2. Initialize OpenVINO Core and Compile for NPU
    core = ov.Core()
    
    # You can change "NPU" to "GPU" (Xe2) or "CPU" to compare performance
    print(f"🚀 Loading model onto Arrow Lake NPU...")
    compiled_model = core.compile_model(model_xml, device_name="NPU")
    output_layer = compiled_model.output(0)

    # 3. Preprocess the Image (Must match training exactly)
    # Read as Grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"❌ Failed to load image at {image_path}")
        return None
    
    # Resize to config dimensions
    image = cv2.resize(image, (WIDTH, HEIGHT))
    
    # Normalize and convert to float32
    # training used transforms.Normalize((0.5,), (0.5,)) -> (x - 0.5) / 0.5
    input_data = image.astype(np.float32) / 255.0
    input_data = (input_data - 0.5) / 0.5
    
    # Add Batch and Channel dimensions: [1, 1, H, W]
    input_data = input_data[np.newaxis, np.newaxis, :, :]

    # 4. Perform Inference
    import time
    start_time = time.perf_counter()
    results = compiled_model(input_data)[output_layer]
    end_time = time.perf_counter()

    # 5. Decode Output
    # The output is [1, 6, 62] -> [Batch, Length, CharSet]
    predicted_indices = np.argmax(results[0], axis=1)
    predicted_text = "".join([CHARS[i] for i in predicted_indices])

    print("-" * 30)
    print(f"🧩 Predicted Captcha: {predicted_text}")
    print(f"⚡ Inference Time: {(end_time - start_time) * 1000:.2f} ms")
    print("-" * 30)
    
    return predicted_text

if __name__ == "__main__":
    # Test with a specific image path
    # Example: python solve-captcha.py test_image.png
    import sys
    if len(sys.argv) > 1:
        solve_captcha(sys.argv[1])
    else:
        print("💡 Usage: python solve-captcha.py <path_to_image>")
