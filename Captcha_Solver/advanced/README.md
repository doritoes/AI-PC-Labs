# Advanced CAPTCHA
Moving into an Advanced tier is a significant jump. We are shifting from a simple 10-digit classifier to a 62-character alphanumeric model (a-z, A-Z, 0-9). We are moving to a 6-character string instead of 4. This increases the complexity of the output layer from 40 neurons to 372 neurons (62 x 6 characters).

To handle this, we will leverage the Intel GPU (iGPU) on the Arrow Lake chip using intel_extension_for_pytorch (IPEX)
- Much faster for training than the CPU alone (iGPU has hundreds of execution units/EUs compare to 20 CPU threads)
- Use CPU to generate training data images, and allow iGPU to train the model
  - This means infinite training data! We use half the CPU cores at the start of the epoch to generate fresh CAPTHAS
- Includes learning speed management (monitors validation loss, when to slow down and fine type, and stops early to prevent over-training)
- `train.py` is cable of resuming training on the results of the last epoch
- Increasing game to solve 100 in 10 seconds
- Disable Intel Wolf Security protections during the training run

Heavily tuned for this hardware
- Tuned dataset size and batch size
  - save VRAM for the 16GB RAM (shared GPU/CPU/system)
  - adjust workload to fit within 35W power envelope
- In-loop memory purge runs to try and avoid memory stall
- Single bank of RAM, so tune RAM load
- NOTE If an epoch is running overly long, heard-clearing GPU cache will bring it back to life
  - `python -c "import torch; torch.xpu.empty_cache(); print('✅ XPU Cache Cleared')"`
  - You need to do this often, but the quality model was worth it!

WARNINGS :warning:
- :warning: this puts an intensive load on your Mini
  - put your Mini on a wire rack and point a fan at it, or similarly keep it cool
- :warning: You are entering dependency hell. The following was tested on this specific environment at the time of writing.
- :warning: This is running on the very edge of RAM. You NEED to close out all apps when you are training.
  - If memory utilization goes over 91%, find something to close
  - If the Epoch time goes too high, check memory
  - If you there is a lot of SSD disk activity, you are swapping memory. Reduce your RAM usage.
  - See [Optimize for Training](Optimize_For_Training.md) for steps to clean up
    - First debloat Windows 11
    - Stop and Disable SysMain service
    - Disable Wolf Security features, or better yet uninstall them following the specific order provided
    - Disable desktop transparency and animation effects (significantly reduces the dwm.exe (Desktop Window Manager) RAM footprint)
      - Search for "View advanced system settings"
      - Under Performance > Settings, choose Adjust for best performance

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
7. `pip install tqdm`
8. `pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/xpu`
9. Confirm the GPU is ready
    - `python -c "import torch; print('XPU Available:', torch.xpu.is_available()); print('Device Name:', torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'Not Found')"`
10. Create a folder named `advanced`
11. In the folder create
    - [config.py](config.py)
    - [model.py](model.py)
    - [train.py](train.py)
    - [convert.py](convert.py)
    - [quantize.py](quantize.py)
    - [solve-captcha.py](solve-captcha.py)
    - [game.py](game.py)
12. Start the GPU cache purge loop in a separate powershell window
    - `python -c "import torch; torch.xpu.empty_cache(); print('✅ XPU Cache Cleared')"`
    - This does a hard flush of GPU cache every minute for 1 day
    - This is CPU to keep the training script working in such a small 16GB memory space
13. Train
    - `python train.py`
    - Key: Hybrid training adding in fresh images to prevent over-fitting on the noise
    - In testing, achieved 92.33% character-level accuracy and 67.00% full solve rate
    - Phases
        - Foundation (Epochs 1-30) First learns the numeric digits
        - Alphanumeric (Epochs 31-70 Add in the rest of the characters while leveraging label smoothing
        - Refinement (Epochs 71-100) Add "perspective" augmentation and improve accuracy
    - Test progress
        - `predict.py` will test the model
        - `benchmark.py` will test the model some more
14. Refine
    - Decided to do additional training to improve on the "hard characters" to identify
    - 5 more epochs
    - `refine.py`
    - Test progress
        - `predict.py` will test the model
        - `benchmark.py` will test the model some more
15. Convert and Quantize
    - `python convert.py`
        - Will see some warnings, it is OK
    - `python quantize.py`
        - Windows Smart App Control (SAC) flags `torch_xpu_ops_aten.dll` as a risk
            - Step 1 - The "Local Exclusion" (Command Line)
                - Open administrative PowerShell
                - `Add-MpPreference -ExclusionPath "C:\Users\sethh\Captcha-AI\"`
            - Step 2 - Force "Unblock" on the entire Env
                - `dir -Path "C:\Users\sethh\Captcha-AI\" -Recurse | Unblock-File`
            - Step 3 - "Developer Mode" Toggle (relaxes SAC policy for unsigned local DLLs)
                - Press Win + I to open Settings
                - Search **User developer features** and open
                - Toggle Developer Mode to **On**
                - Restart your PC. This is crucial because SAC policies are often loaded at boot.
        - Will see errors, is is OK
16. Test
    - `python solve-captcha.py`
    - `python game.py`
