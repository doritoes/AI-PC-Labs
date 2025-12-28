# Captcha Solver Lab
The mission in this Lab is to create genuine hands-on experience for technology learners using the learner-friendly NUC PCs. This is a second Lab to demonstrate the NPU boost for Captcha solving.

NOTE Intel NPUs currently perform best when accessed directly from Windows. The NPU driver support inside the Linux kernel for WSL is still maturing.

IMPORTANT Be aware that the drivers and models and versions are very fluid, and what works today might not working tomorrow!
- To get the chatbot to work, updated drivers were required that support OpenVINO
- Ran into dynamic shape error, because the NPU driver (v4545) requires a "fixed map" of the math it is about to perform
- Had to bypass the NPUW optimizer because it currently lacks a "map" for the specific architecture of Qwen2.5 lm_head layer, leading to a "MatMul map" crash
- Level Zero (L0) Hardware Abstraction was disabled, forced OpenVINO to use a more stable communication path to the silicon, preventing memory sync errors between the system RAM and the NPU's local scratchpad. The dedicated L0 memory was unable to be reserved.

Prerequisites:
- Window 11 Pro already installed and configured

Mission:
- Set up Python environment
- Install OpenVINO
- Verify Hardware
- Select Model
- Power Test
- Future Projects

Materials:
- Windows 11 PC with Intel Core Ultra  processor (dedicated NPU)

# Overview
## Day 1 - Model Setup and Training
1. Environment setup - Install Intell OpenVINO Toolkit and PyTorch
    - [Python](1_Python.md)
    - [OpenVINO](2_OpenVINO.md)
    - [PyTorch](3_PyTorch.md)
2. Train model (prepare dataset, model architecture; run training)
    [Train Model](4_Train_Model.md)

## Day 2 - NPU Deployment and "The Hack"
The model is currently using FP32 (32-bit floating point) math. The NPU on the Core Ultra is built for INT8 (8-bit integer) math. When we convert and quantize the mode, it is "rounding" the AI's logic to make it run faster on specialized hardware.

1. Convert from PyTorch to OpenVINO
    - [Convert to OpenVINO](5_Convert_OpenVINO.md)
    - WTH it looks like it's retraining!
2. Quantization
    - [Quantize](6_Quantize.md)
4. Model Conversion - Convert the trained model to the OpenVINO Intermediate Representation (IR) format
5. Quantization (Crucial Step): Use OpenVINO’s NNCF to quantize the model to INT8
    - The NPU (13 TOPS) performs best with INT8 math. Quantization reduces model size and increases speed with minimal accuracy loss.
6. Inference on NPU - Write a script using core.compile_model(model, "NPU")
    - Observe the "Task Manager > Performance" tab to see the NPU usage spike while the CPU remains idle
7. Cybersecurity Lab Task - Create a "Brute Force Simulator" where the script must solve 100 CAPTCHAs in under 5 seconds to "bypass" a mock login portal

## Cleanup
Steps required to clean up after the Lab is complete.

~~~
# 1. Deactivate the environment if it is currently active
if ($env:VIRTUAL_ENV) { deactivate }

# 2. Remove the main project directory and all contents (Models, Env, Scripts)
$ProjectDir = "$env:USERPROFILE\Edge-AI"
if (Test-Path $ProjectDir) {
    Write-Host "Removing Project Directory: $ProjectDir" -ForegroundColor Yellow
    Remove-Item -Path $ProjectDir -Recurse -Force
}

# 3. Remove the hidden Hugging Face cache (The heavy 'blob' files)
$HFCache = "$env:USERPROFILE\.cache\huggingface"
if (Test-Path $HFCache) {
    Write-Host "Removing Hugging Face Cache: $HFCache" -ForegroundColor Yellow
    Remove-Item -Path $HFCache -Recurse -Force
}

# 4. Remove OpenVINO's default internal cache (if any was created)
$OVConfig = "$env:USERPROFILE\AppData\Local\OpenVINO"
if (Test-Path $OVConfig) {
    Remove-Item -Path $OVConfig -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "Cleanup Complete. All Edge-AI components removed." -ForegroundColor Green
~~~
