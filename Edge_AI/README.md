# Edge AI Lab
The mission in this Lab is to create genuine hands-on experience for technology learners using the learner-friendly NUC PCs. This is a good "first AI Lab" that demonstrates the power of the NPU. We will compare NPU vs. CPU performance.

NOTE Intel NPUs currently perform best when accessed directly from Windows. The NPU driver support inside the Linux kernel for WSL is still maturing.

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
- A webcam to capture video

# Overview
## Hardware and Driver Verification
Ensure Windows kernal can "see" the AI hardware
- Device Manager > Neural Processors
  - You should see Intel(R) AI Boost
- Task Manager > Performance
  - Look for NPU 0 in the sidebar
  
## Set up Python
In this step you will set up the Python environment.

[Python](1_Python.md)

## Install OpenVINO
In this step you will install Intel OpenVINO

[OpenVINO](2_OpenVINO.md)

## Lab 1 - NPU Demonstration
Our first Lab demonstrating the Edge AI will demonstrate and benchmark AI workloads running at the CPU, GPU and NPU.

[Image Classification](3_Image_Classification.md)


## Lab 2 - Privacy-First Edge Vision
Privacy-First Edge Vision (Object Detection)

[Privacy-First Edge Vision](4_Privacy_Edge_Vision.md)

Goal: Build a "Smart Surveillance" application that detects objects (people, vehicles) in a video stream using only local hardware.

Process: Use a YOLOv8 (You Only Look Once) model. Students will write a Python script using opencv to capture a webcam feed or video file and send frames to the NPU for processing.

Measurement: "Inference time per frame" (ms).

Key Learning: "Double Buffering." Students learn how the CPU handles the "Pre-processing" (resizing images) and "Post-processing" (drawing boxes) while the NPU handles the "Inference" (the actual math) in parallel.

Deliverable: A live video window with real-time bounding boxes running smoothly on the NPU.

## Lab 3 - Local Chatbot (OpenVINO GenAI)
The Local Chatbot (OpenVINO GenAI)

[Local Chatbot](5_Local_Chatbot.md)

Goal: Run a Large Language Model (LLM) like Phi-3 or Llama-3 entirely offline.

Process: Use the openvino-genai library (which you installed) to load a quantized (INT4) model.

Measurement: "Tokens per second" and "Time to First Token" (TTFT).

Key Learning: Understanding Quantization. Students learn how to shrink a multi-gigabyte model into a 2GB "weight" file so it fits into the NPU's memory. They will use the BEST_PERF hint to optimize the NPU's power.

Script Snippet: Python

import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline("model_path", "NPU")
print(pipe.generate("Explain Edge AI in three sentences.", max_new_tokens=100))

## Cleanup and Next Steps

Steps required to clean up after the Lab is complete

Additional project ideas
1. Project 1: The Privacy-First Smart Camera
    - Goal: Build a real-time object detection system that blurs human faces locally before any data is saved or streamed.
    - Why NPU: It demonstrates real-time processing without slowing down the rest of the OS.
2. Project 2: Local LLM Chatbot
    - Goal: Use OpenVINO GenAI to run a small Large Language Model (like Llama 3 or Phi-3) entirely offline.
    - Why NPU: Students can see how the NPU handles the "thinking" (text generation) while the GPU remains free for the UI or other tasks.
3. Project 3: AI-Driven "System Health" Monitor
    - Goal: Create a script that uses a Time-Series AI model to predict when the homelab server might overheat based on fan speeds and workload.
    - Why NPU: This is a classic "background task" that should run 24/7 without wasting the PC's main power.
