# Install OpenVINO
In this step we install the bridge between Python and your Intel NPU.

## Install
1. Ensure the Python virtual environment is active.
    - `.\ai_lab_env\Scripts\Activate`
2. Update and upgrade pop
    - `python -m pip install --upgrade pip
3. Install OpenVINO and essential AI PC libraries
    - `pip install openvino==2025.4.1 openvino-genai==2025.4.1 opencv-python numpy`
    - NOTE 'openvino-genai' is specifically designed for LLMs on the NPU
  
## Test
1. Create the NPU verification script `verify_npu.py`
~~~
import openvino as ov
import numpy as np
import sys

def verify_npu_lab():
    print("--- OpenVINO 2025 NPU Verification ---")
    
    # Initialize OpenVINO Core
    core = ov.Core()
    
    # 1. Hardware Check
    devices = core.available_devices
    print(f"Detected Devices: {devices}")
    
    if 'NPU' not in devices:
        print("❌ ERROR: Intel AI Boost NPU not detected.")
        print("Check if 'Intel(R) AI Boost' is visible in Windows Device Manager.")
        sys.exit(1)

    # 2. Extract NPU Metadata
    npu_name = core.get_property("NPU", "FULL_DEVICE_NAME")
    print(f"✅ NPU Confirmed: {npu_name}")

    # 3. Functional Test: Model Compilation
    # We create a simple 'Identity' model to test the driver communication path
    print("🔄 Testing NPU Driver (Level Zero) by compiling a test model...")
    
    # Define a simple 1x224x224 input (standard image size)
    input_node = ov.runtime.opset13.parameter([1, 3, 224, 224], np.float32, name="input")
    model = ov.Model(ov.runtime.opset13.relu(input_node), [input_node], "TestModel")
    
    try:
        # This is where the magic happens: offloading to NPU
        compiled_model = core.compile_model(model, "NPU")
        print("🚀 Success! The NPU has successfully compiled the test model.")
        
        # 4. Quick Inference Test
        test_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
        infer_request = compiled_model.create_infer_request()
        infer_request.infer({input_node.any_name: test_input})
        
        print("✨ Inference complete. Your HP EliteDesk NPU is 100% ready for the lab.")
        
    except Exception as e:
        print(f"❌ COMPILATION ERROR: {e}")
        print("\nTip: Ensure your Intel NPU drivers are updated to version 32.0.100.x or higher.")

if __name__ == "__main__":
    verify_npu_lab()
~~~

2. Run the script: `python3 verify_npu.py`
