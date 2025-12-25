# AI-PC-Labs
My original Labs were centered around the Intel NUC. The newer Intel® Core™ Ultra processors add a dedicated NPU (neural processing unit) to enabled the feature called "Intel AI Boost".

These labs are designed to demonstrate practice AI skills leveraging Intel AI Boost.
- built on HP EliteDesk 8 Mini G1i Desktop AI PC
  - Windows 11 Pro
  - Intel® Core™ Ultra 7 265T (up to 5.3 GHz with Intel® Turbo Boost Technology, 30 MB L3 cache, 20 cores, 20 threads)
  - 16 GB memory
  - 256 GB SSD storage Intel® Graphics
  - NPU 13 TOPS
  - With CPU and GPU total 33TOPS

# Why
1. Edge AI is the deployment of machine learning models directly onto local hardware devices (the Mini G1i) rather than on centralized cloud servers. The "Edge" refers to the fact that the AI processing happens at the "edge" of the network, right where the data is being generated.
2. Using the NPU we have some big advantages
    - Designed for sustained low-power AI tasks (background tasks like background blur in video, real-time noise cancelation, etc) with minimal impact on battery or thermals
    - Low latency (great for real-time apps like robotics or video analytics)
    - Bandwidth efficiency (only send the final "inference" over the Internet, if necessary)
    - Privacy and security (facial recognition, voice processing, data analysis stays in the lab)
    - Offloading

# How
1. The stack
    - Hardware: Intel AI Boost (NPI)
    - Optimization: OpenVINO Toolkit (coverts standard models into a format the NPU understands)
        - acts as a bridge between AI models (PyTorch, TensorFlow) and the NPU hardware
    - Inference: Running the optimized model locally using Python or C++

# Labs
Hands-on learning projects that provide exposure to multiple advanced technologies centered around an NPU-powered PC.

1. Edge AI 1 ▶️ [Get Started!](Edge_AI/README.md)
