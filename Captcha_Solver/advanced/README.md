# Advanced CAPTCHA
Moving into an Advanced tier is a significant jump. We are shifting from a simple 10-digit classifier to a 62-character alphanumeric model ($a-z, A-Z, 0-9$). We are moving to a 6-character string instead of 4. This increases the complexity of the output layer from 40 neurons to 372 neurons (62 x 6 characters).

To handle this, we will leverage the Intel GPU (iGPU) on the Arrow Lake chip using intel_extension_for_pytorch (IPEX)
- much faster for training than the CPU alone (iGPU has hundreds of execution units/EUs compare to 20 CPU threads)
- use CPU to generate training data images, and allow iGPU to train the model
- includes early stopping (monitors validation loss and stops early to prevent over-training)
- uses 40,000 images for training
- increasing game to solve 100 in 10 seconds
- Disable Intel Wolf Secruity protections during the training run

Heavily tuned for this hardware
- tuned dataset size and batch size
  - save VRAM for the 16GB RAM (shared GPU/CPU/system)
  - adjust workload to fit within 35W power envelope
- in-loop memory purge runs every 500 bathes to try and avoid memory stall
- single bank of RAM, so tune RAM load
- If an epoch is running long, heard-clearing GPU cache will bring it back to life
  - `python -c "import torch; torch.xpu.empty_cache(); print('✅ XPU Cache Cleared')"`

WARNING this puts an intensive load on your GPU
- put your Mini on a wire rack and point a fan at it, or similarly keep it cool

WARNING you are entering dependency hell. The following was tested on this specific environment at the time of writing.

WARNING this is running on the very edge of RAM. You NEED to close out all apps when you are training. If memory utilization goes over 91%, find something to close. If the Eposh time goes too high, check memory. If you there is a lot of SSD disk activity, you are swapping memory. Reduce your RAM usage. 

## Advanced CAPTCHA Model
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
5. `python -m pip install --upgrade pip setuptools wheel`
6. `pip install opencv-python pillow captcha onnx openvino==2025.4.1 numpy nncf`
7. `pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/xpu`
8. Confirm the GPU is ready
    - `python -c "import torch; print('XPU Available:', torch.xpu.is_available()); print('Device Name:', torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'Not Found')"`
9. Create a folder named `advanced`
10. In the folder create
    - [config.py](config.py)
    - [train.py](train.py)
    - [convert.py](convert.py)
    - [quantize.py](quantize.py)
    - [solve-captcha.py](solve-captcha.py)
    - [game.py](game.py)
11. Train
     - `python train.py`
     - Let it cook! The iGPU is a beast.
     - Phases
         - Foundation (Epochs 1-5) Loss drops steadily. Accuracy stays at 0% or very low (<1%)
         - Breakthough (Epochs 6-15) Accuracy begins to "pop" (e.g., 5% > 20% > 50%)
         - Refinement (Epochs 16-30) Loss flattens. Accuracy climbs toward 90%+
12. Convert and Quantize
     - `python convert.py`
     - `python quantize.py`
13. Test
    - `python solve-captcha.py`
    - `python game.py`
