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
1. Model Conversion - Convert the trained model to the OpenVINO Intermediate Representation (IR) format
2. Quantization (Crucial Step): Use OpenVINO’s NNCF to quantize the model to INT8
    - The NPU (13 TOPS) performs best with INT8 math. Quantization reduces model size and increases speed with minimal accuracy loss.
3. Inference on NPU - Write a script using core.compile_model(model, "NPU")
    - Observe the "Task Manager > Performance" tab to see the NPU usage spike while the CPU remains idle
4. Cybersecurity Lab Task - Create a "Brute Force Simulator" where the script must solve 100 CAPTCHAs in under 5 seconds to "bypass" a mock login portal

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

## Final Thoughts
1. AI Hardware is Not "One Size Fits All" - Code written for a CPU does not automatically work on an NPU.
    - Learning: CPUs are Generalists (flexible but power-hungry), while NPUs are Specialists (rigid but highly efficient).
    - Evidence: The NPU required a specific "Compilation" step to map the model's math to its physical transistors—something a CPU's "Interpreter" approach skips.
2. The "Static Shape" Constraint - The most significant hurdle was the NPU's inability to handle dynamic data sizes.
    - The Learning: High-efficiency silicon often requires Deterministic Memory. We had to tell the hardware exactly how much space to reserve for the prompt (MAX_PROMPT_LEN) and the response (MIN_RESPONSE_LEN).
    - Insight: In the world of NPUs, "flexibility" is often traded for "predictability." If the data doesn't fit the "train tracks" we laid down, the system crashes.
3. Quantization: The Art of the Compromise
    - Running a massive model like Qwen (1.5 billion parameters) on a laptop was only possible through 4-bit Quantization (INT4)
    - Learning: Shrinking a model makes it small enough to fit in the NPU's local memory, but it introduces "rounding errors"
    - Evidence: Students saw the model correctly identify a math formula but fail the actual arithmetic. This teaches that quantized models should be used for language tasks, not as high-precision calculators.
4. Optimization vs. Stability (The "Bleeding Edge") - The lab required several "workarounds" (like disabling NPUW and L0 environment variables) to achieve stability.
    - Learning: Software drivers and hardware are often out of sync. Developers must sometimes disable "advanced" optimization features to ensure "basic" reliability.
    - Insight: Being an AI Developer means knowing how to look at low-level logs (like the MatMul map error) and adjusting hardware-level flags to bridge the gap between the model and the silicon.

All the workarounds effectively turned a "General Intelligence" model into a Fixed-Function Appliance. For the purpose of the lab, it demonstrated that while NPUs are the future of efficiency, they currently require a "Handshake" between the developer and the hardware—manually defining the boundaries that the hardware is not yet flexible enough to find on its own.
