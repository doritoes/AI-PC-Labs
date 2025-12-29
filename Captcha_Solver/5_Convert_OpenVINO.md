# Convert to OpenVINO
The next step is to create and run a conversion script to move from PyTorch to OpenVINO.

Sometimes a mismatch occurs in the bounds in the quantized model when there is a dynamic input shape to the model. The NPU is much more "picky" about exact dimensions than the CPU is. Therefore we will tell the OpenVINO exactly what the "grid" looks like so the NPU doesn't try to guess.

1. Make sure you ran the `train.py` script in the previous step to create the model
2. Create script from [convert.py](convert.py)
3. `python convert.py`
