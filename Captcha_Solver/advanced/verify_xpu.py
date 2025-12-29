import os
import sys
import ctypes

# 1. Define the paths where the Intel DLLs live
venv_path = sys.prefix
torch_lib = os.path.join(venv_path, "Lib", "site-packages", "torch", "lib")
ipex_bin = os.path.join(venv_path, "Lib", "site-packages", "intel_extension_for_pytorch", "bin")

# 2. Tell Windows to allow loading from these folders
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)
if os.path.exists(ipex_bin):
    os.add_dll_directory(ipex_bin)

# 3. Now try the import
try:
    import torch
    import intel_extension_for_pytorch as ipex
    print(f"✅ Success! Torch version: {torch.__version__}")
    print(f"✅ XPU available: {torch.xpu.is_available()}")
    if torch.xpu.is_available():
        print(f"🚀 GPU Name: {torch.xpu.get_device_name(0)}")
except Exception as e:
    print(f"❌ Still failing: {e}")
