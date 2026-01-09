import os
import cv2
import numpy as np
import openvino as ov
import time
from config import CHARS, CAPTCHA_LENGTH, WIDTH, HEIGHT

def solve_captcha(image_path, model_type="int8"):
    # 1. Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path logic for the 'advanced' folder structure
    if model_type == "int8":
        model_xml = os.path.join(current_dir, "openvino_int8_model", "captcha_model_int8.xml")
    else:
        model_xml = os.path.join(current_dir, "openvino_model", "captcha_model.xml")

    if not os.path.exists(model_xml):
        print(f"❌ Model not found at {model_xml}. Run quantize.py first.")
        return None

    # 2. Initialize OpenVINO Core
    core = ov.Core()
    
    # We use AUTO:NPU,GPU to ensure the script never crashes even if the NPU is busy
    print(f"🚀 Initializing AI Boost NPU...")
    try:
        compiled_model = core.compile_model(model_xml, device_name="AUTO:NPU,GPU")
        device_used = compiled_model.get_property("FULL_DEVICE_NAME")
        print(f"✅ Model loaded on: {device_used}")
    except Exception as e:
        print(f"⚠️ NPU/GPU not available, falling back to CPU. Error: {e}")
        compiled_model = core.compile_model(model_xml, device_name="CPU")

    # 3. Preprocess the Image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"❌ Failed to load image at {image_path}")
        return None
    
    # Resize to the 200x80 dimensions used in the lab
    image = cv2.resize(image, (WIDTH, HEIGHT))
    
    # Advanced Normalization: (x - 0.5) / 0.5
    input_data = image.astype(np.float32) / 255.0
    input_data = (input_data - 0.5) / 0.5
    
    # Add dimensions for OpenVINO: [Batch, Channel, Height, Width]
    input_data = input_data.reshape(1, 1, HEIGHT, WIDTH)

    # 4. Perform Inference
    start_time = time.perf_counter()
    results = compiled_model(input_data)[0]
    latency = (time.perf_counter() - start_time) * 1000

    # 5. Decode with Confidence Calculation
    # Convert raw outputs (logits) into probabilities (0-1)
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    # Reshape to [6 characters, 62 classes]
    predictions = results.reshape(CAPTCHA_LENGTH, -1)
    
    predicted_text = ""
    confidences = []

    for char_logits in predictions:
        probs = softmax(char_logits)
        idx = np.argmax(probs)
        conf = np.max(probs)
        
        predicted_text += CHARS[idx]
        confidences.append(conf)

    avg_confidence = np.mean(confidences) * 100

    print("-" * 40)
    print(f"🧩 Predicted: {predicted_text}")
    print(f"🎯 Confidence: {avg_confidence:.1f}%")
    print(f"⚡ NPU Latency: {latency:.2f} ms")
    print("-" * 40)
    
    return predicted_text, avg_confidence

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        solve_captcha(sys.argv[1])
    else:
        # Quick check: look for any .png in the current folder if no arg given
        print("💡 Usage: python solve-captcha.py <path_to_image>")
