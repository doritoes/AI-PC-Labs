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

def test_npu():
    print("--- AI PC Hardware Verification ---")
    core = ov.Core()
    devices = core.available_devices

    # 1. Check if NPU exists in the device list
    if 'NPU' not in devices:
        print("❌ ERROR: NPU not found in available devices.")
        print(f"Devices detected: {devices}")
        return

    # 2. Get detailed NPU Information
    full_name = core.get_property("NPU", "FULL_DEVICE_NAME")
    print(f"✅ SUCCESS: Found {full_name}")

    # 3. Test "Warm-up" (Compile a dummy model to NPU)
    # This checks if the Level Zero drivers are working
    print("🔄 Compiling test model to NPU... (this may take a few seconds)")
    
    # Create a simple 1x10 matrix addition model
    param = ov.runtime.opset10.parameter([1, 10], name="input", dtype=np.float32)
    relu = ov.runtime.opset10.relu(param)
    model = ov.Model([relu], [param], "VerifyNPU")
    
    try:
        compiled_model = core.compile_model(model, "NPU")
        print("🚀 NPU model compilation successful!")
        
        # 4. Run a test inference
        input_data = np.random.rand(1, 10).astype(np.float32)
        results = compiled_model(input_data)
        print("✨ NPU Inference test complete. Hardware is fully operational.")
        
    except Exception as e:
        print(f"❌ COMPILATION ERROR: Drivers may be missing or mismatched.\n{e}")

if __name__ == "__main__":
    test_npu()
~~~

2. Run the script: `python3 verify_npu.py`
