# Install PyTorch
In this step we install PyTorch, the primary engine we will use to build and train our CAPTCHA-solving neural network. We will also install specialized libraries for image generation and model optimization.

## Install
1. Ensure the Python virtual environment is active
    - Open a <in>new PowerShell window</ins>
    - `cd "$env:USERPROFILE\Captcha-AI"`
    - `.\nputest_env\Scripts\Activate`
2. Install PyTorch (CPU-Optimized)
    - Since the Intel NPU handles the inference (the "solving"), we will use the CPU-optimized version of PyTorch for the initial training
    - `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
3. Install Training & Security Dependencies
    - `pip install matplotlib numpy opencv-python captcha nncf`
        - `captcha` - used to create realistic, distorted text images (security challenges)
        - `nncf` - Neural Network Compression Framework - secret sauce to shrink the model for the NPU
## Test
1. Hardware Check
    - Run the script ot verify PyTorch is ready to use all 20 cores of the Intel Core Ultra 7 265T
        - `python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'Available CPU Threads: {torch.get_num_threads()}')"`
        - Expect to see PyTorch Version: 2.9.1+cpu and Available CPU Threads: 20

## Learn More
In this Lab we are using the CPU to train, allowing the model to "learn" You can also use the GPU to train.

The Intel Arc Graphics (iGPU) in the Core Ultra can actually be used for training via `intel-extension-for-pytorch` or `torch-directml`

We stick to the CPU for this lab because training a small CAPTCHA model on 20CPU threads is already fast (under 5-10 minutes). Adding the GPU drivers and "XPU" (Intel's GPU backend for PyTorch) can take longer to set up than the time it saves.

To see the GPU in action, install IPEX (Intel Extension for PyTorch). Use `xpu` device instead of `cpu`.

Example code:
```
# Standard PyTorch
device = torch.device("cpu") 

# Intel GPU Optimized (Requires ipex)
import intel_extension_for_pytorch as ipex
device = torch.device("xpu") 

model.to(device)
```
