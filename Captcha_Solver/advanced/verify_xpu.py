import os
import sys

# 1. Define base paths
venv_path = sys.prefix
torch_lib = os.path.join(venv_path, "Lib", "site-packages", "torch", "lib")
# NEW: Specifically add the IPEX bin directory which contains the XPU kernels
ipex_bin = os.path.join(venv_path, "Lib", "site-packages", "intel_extension_for_pytorch", "bin")

# 2. Add directories to DLL search path (CRITICAL FOR WINDOWS)
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)
    print(f"[*] Added to DLL path: {torch_lib}")

if os.path.exists(ipex_bin):
    os.add_dll_directory(ipex_bin)
    print(f"[*] Added to DLL path: {ipex_bin}")

# 3. Import and Test
try:
    import torch
    import intel_extension_for_pytorch as ipex
    print(f"✅ Success! Torch version: {torch.__version__}")
    print(f"✅ XPU available: {torch.xpu.is_available()}")
    if torch.xpu.is_available():
        print(f"🚀 GPU Name: {torch.xpu.get_device_name(0)}")
except Exception as e:
    print(f"❌ Still failing: {e}")
