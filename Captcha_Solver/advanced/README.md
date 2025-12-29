# Advanced CAPTCHA
Moving into an Advanced tier is a significant jump. We are shifting from a simple 10-digit classifier to a 62-character alphanumeric model ($a-z, A-Z, 0-9$). We are moving to a 6-character string instead of 4. This increases the complexity of the output layer from 40 neurons to 372 neurons (62 x 6 characters).

To handle this, we will leveage the Intel GPU (iGPU) on your Arrow Lake chip using intel_extension_for_pytorch (IPEX)
- much faster for training than the CPU alone (iGPU has hundreds of execution units/EUs compare to 20 CPU threads)
- includes early stopping (monitors validation loss and stops if it plateaus for 3 epochs) and XPU Optimization
- uses 50,000 images + 50,000 slightly modified images for training (1:1 augmentation)
- use CPU to generate training data images, and allow iGPU to train the model simultaneously
- increasing game to solve 100 in 10 seconds
- uses 5-8GB of disk for generating training data

WARNING you are entering dependency hell. The following was tested on this specific environment.

NOTE Once the model is at 90% accuracy you can increase the complexity of the augmentations like adding noise or lines

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
6. `pip install opencv-python pillow captcha onnx openvino==2025.4.1 numpy`
7. `pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/xpu`
8. Confirm the GPU is ready
    - `python -c "import torch; print('XPU Available:', torch.xpu.is_available()); print('Device Name:', torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'Not Found')"`
9. Create a folder named `advanced`
10. In the folder create
    - [config.py](config.py)
    - [train.py](train.py)
    - [convert.py](convert.py)
    - [quantize.py](quantize.py)
    - solve-captcha.py
    - [game.py](game.py)
11. Train
     - `python train.py`
12. Convert and Quantize
     - `python convert.py`
     - `python quantize.py`
13. Test
    - `python solve-captcha.py`
    - `python game.py`
