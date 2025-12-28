# Quantize the Model
Quantization reduces model size and increases speed with minimal accuracy loss. To unlock the 13 TOPS NPU, we need to apply Post-Training Quantization. This is where OpenVINO looks at some of your "held-back" images to figure out how to compress the model without losing that 74% accuracy. The quantize step is "rounding" the AI's logic to make it run faster on the specialized NPU.

The model is currently using FP32 (32-bit floating point) math. The NPU on the Core Ultra is built for INT8 (8-bit integer) math. Let's run the script to do this.


# Convert to OpenVINO
The next step is to create and run a conversion script to move from PyTorch to OpenVINO.

1. Make sure you ran the `train.py` and `convert.py` scripts in the previous steps
2. Create script from [quantize.py](quantize.py)
3. `python quantize.py`
