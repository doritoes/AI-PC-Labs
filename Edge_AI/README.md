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

# Overview
## Hardware and Driver Verification
Ensure Windows kernal can "see" the AI hardware
- Device Manager > Neural Processors
  - You should see Intel(R) AI Boost
- Task Manager > Performance
  - Loof for NPU 0 in the sidebar
  
## Set up Python
In this step you will set up the Python environment.

[Python](1_Python.md)

1. Install Python
Download Python 3.10 or 3.11 from python.org.

Crucial: During installation, check the box "Add Python to PATH".

## Install OpenVINO
In this step you will install Intel OpenVINO

[OpenVINO](2_OpenVINO.md)

## Lab 1 - NPU Demonstration
Our first Lab demonstrating the Edge AI.


## Lab 2

## Lab 3


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
