# Advanced CAPTCHA
Moving into an Advanced tier is a significant jump. We are shifting from a simple 10-digit classifier to a 62-character alphanumeric model ($a-z, A-Z, 0-9$). We are moving to a 6-character string instead of 4. This increases the complexity of the output layer from 40 neurons to 372 neurons (62 x 6 characters).

To handle this, we will leveage the Intel GPU (iGPU) on your Arrow Lake chip using intel_extension_for_pytorch (IPEX)
- much faster for training than the CPU alone (iGPU has hundreds of execution units/EUs compare to 20 CPU threads)
- includes early stopping (monitors validation loss and stops if it plateaus for 3 epochs) and XPU Optimization
- uses 50,000 images + 50,000 slightly modified images for training
- increasing game to solve 100 in 10 seconds

1. If you haven't already, install Python 3.12
    - https://www.python.org/downloads/windows/
    - Use: Python 3.12.10
    - Download Windows installer (64-bit) https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe
2. Exit the Python 3.14 environment if active
    - `deactivate`
3. Create new environment for Python 3.12
    - `py -3.12 -m venv nputest_advanced_env`
4. Activate the new environment
    - `.\nputest_advanced_env\Scripts\activate`
    - `python --version`
5. Install Intel Extension for PyTorch
    - `python -m pip install --upgrade pip`
    - `pip install intel-extension-for-pytorch`
    - `python -m pip install torch==2.5.1+cxx11.abi torchvision==0.20.1+cxx11.abi torchaudio==2.5.1+cxx11.abi intel-extension-for-pytorch==2.5.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/`
    - `pip install intel-openmp==2025.0.4 dpcpp-cpp-rt==2025.0.4 mkl-dpcpp==2025.0.1`
    - Copy the "missing" Intel DLLs to where PyTorch expects them
      - Create [fix_intel_gpu.ps1](fix_intel_gpu.ps1)
      - Run: `.\fix_intel_gpu.ps1`
  - Test:
      - Create [verify_xpu.py](verify_xpu.py)
      - `.\verify_xpu.py`
2. Add to path
    - $env:PATH += ";C:\Users\sethh\Captcha-AI\nputest_advanced_env\Lib\site-packages\intel_extension_for_pytorch\bin"
    - $env:PATH += ";C:\Users\sethh\Captcha-AI\nputest_advanced_env\Lib\site-packages\torch\lib"
6. Confirm the GPU is ready
    - `python -c "import torch; import intel_extension_for_pytorch as ipex; print(f'XPU Available: {torch.xpu.is_available()}');
7. Run verify_xpu.py
4. Create a folder named `advanced`
print(f'GPU Name: {torch.xpu.get_device_name(0)}')"`
4. In the folder create
    - [config.py](config.py)
    - [train.py](train.py)
    - [convert.py](convert.py)
    - quantize.py
    - solve-captcha.py
    - [game.py](game.py)
5. Train
     - `python train.py`
6. Convert and Quantize
     - `python convert.py`
     - `python quantize.py`
7. Test
    - `python solve-captcha.py`
    - `python game.py`
