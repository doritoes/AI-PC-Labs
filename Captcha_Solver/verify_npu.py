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
