import os
import cv2
import numpy as np
import openvino as ov
import time
import config # Ensure your config.py has WIDTH=200, HEIGHT=80, and the 62 CHARS

def solve_captcha(image_path, model_type="int8"):
    # 1. Setup Paths (Updated for your 'advanced' folder structure)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if model_type == "int8":
        model_xml = os.path.join(current_dir, "openvino_int8_model", "captcha_model_int8.xml")
    else:
        # Standard FP16/FP32 model directory
        model_xml = os.path.join(current_dir, "openvino_model", "captcha_model.xml")

    if not os.path.exists(model_xml):
        print(f"❌ Model not found: {model_xml}")
        return None

    # 2. Initialize OpenVINO Core
    core = ov.Core()
    
    # Using AUTO:NPU,GPU allows the game script to be resilient
    print(f"🚀 Initializing AI Boost NPU...")
    compiled_model = core.compile_model(model_xml, device_name="AUTO:NPU,GPU")
    infer_request = compiled_model.create_infer_request()

    # 3. Preprocess the Image
    # Read as Grayscale to match the Lab training
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"❌ Failed to load image at {image_path}")
        return None
    
    # Resize to config dimensions (200x80)
    image = cv2.resize(image, (config.WIDTH, config.HEIGHT))
    
    # Normalization: x = (x - 0.5) / 0.5
    input_data = image.astype(np.float32) / 255.0
    input_data = (input_data - 0.5) / 0.5
    
    # Reshape for OpenVINO [Batch, Channel, Height, Width]
    input_data = input_data.reshape(1, 1, config.HEIGHT, config.WIDTH)

    # 4. Perform Inference
    start_time = time.perf_counter()
    results = compiled_model(input_data)[0] 
    latency = (time.perf_counter() - start_time) * 1000

    # 5. Decode and Calculate Confidence
    # The model outputs raw 'logits'. We use Softmax to get probabilities.
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    # Reshape output to [6 characters, 62 possible classes]
    predictions = results.reshape(config.CAPTCHA_LENGTH, -1)
    
    decoded_text = ""
    confidences = []

    for char_logits in predictions:
        probs = softmax(char_logits)
        char_index = np.argmax(probs)
        confidence = np.max(probs)
        
        decoded_text += config.CHARS[char_index]
        confidences.append(confidence)

    avg_confidence = np.mean(confidences) * 100

    print("-" * 40)
    print(f"🧩 Result: {decoded_text}")
    print(f"🎯 Confidence: {avg_confidence:.1f}%")
    print(f"⚡ Speed: {latency:.2f} ms")
    print("-" * 40)
    
    return decoded_text, avg_confidence

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        solve_captcha(sys.argv[1])
    else:
        print("💡 Usage: python solve-captcha.py <path_to_image>")
