# Install OpenVINO
In this step we install the bridge between Python and your Intel NPU.

NOTE Since the Intel Core Ultra 7 265T is part of the newer Arrow Lake family, ensure you specify openvino>=2024.5 to guarantee support for the latest NPU features.

## Install
1. Install the C++ Redistributable
    - NOTE This allows building from source in below steps
    - https://visualstudio.microsoft.com/downloads/?q=build+tools
    - Scroll down to **All Downloads** and locate **Build Tools for Visual Studio 2026**, click **Download**
    - Launch the installer named similar to **VS_BuildTools.exe**
    - Check the box and click **Install**
    - Select the Workload, **Desktop development with C++**
    - Click **Install**
    - When the installation is complete, close the installer
    - Reboot; the build commands are now in the PATH
2. Ensure the Python virtual environment is active
    - Open a <in>new PowerShell window</ins>
    - `cd "$env:USERPROFILE\Edge-AI"`
    - `.\nputest_env\Scripts\Activate`
3. Update and upgrade pip
    - `python -m pip install --upgrade pip`
4. Install OpenVINO and essential AI PC libraries
    - `pip install numpy`
    - `pip install openvino==2025.4.1`
    - `pip install openvino-genai==2025.4.1`
        - NOTE 'openvino-genai' is specifically designed for LLMs on the NPU
    - `pip install opencv-python`

## Test
1. Test if NPU is visible to OpenVINO
    - `python -c "import openvino as ov; core = ov.Core(); print(core.available_devices)"`
    - Should see `['CPU', 'GPU', 'NPU']`
2. Create the NPU verification script `verify_npu.py`
~~~
import openvino as ov
import numpy as np
import sys

def verify_npu_lab():
    print("--- OpenVINO 2025 NPU Verification ---")
    
    # 1. Initialize Core (New 2025 namespace)
    core = ov.Core()
    
    devices = core.available_devices
    if 'NPU' not in devices:
        print(f"❌ ERROR: NPU not detected. Found: {devices}")
        sys.exit(1)

    print(f"✅ NPU Confirmed: {core.get_property('NPU', 'FULL_DEVICE_NAME')}")

    # 2. Create a simple model
    input_node = ov.opset13.parameter([1, 3, 224, 224], np.float32, name="input")
    model = ov.Model(ov.opset13.relu(input_node), [input_node], "VerificationModel")
    
    try:
        print("🔄 Compiling test model to NPU...")
        compiled_model = core.compile_model(model, "NPU")
        print("🚀 Success! Model compiled to NPU.")
        
        # 3. Inference Test
        # In 2025, we use the input/output port objects directly as keys
        test_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
        
        # Method: Run inference directly on the compiled model
        results = compiled_model(test_input)
        
        # Access results using the output port index
        print("✨ Inference complete. NPU is 100% operational.")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    verify_npu_lab()
~~~

3. Run verification script
    - `python verify_npu.py`
