# Image Classification
Our first Lab demonstrating the Edge AI will demonstrate and benchmark AI workloads running at the CPU, GPU and NPU.

Goal: Prove that the NPU can handle heavy AI workloads while leaving the CPU and GPU completely idle for other tasks.

Process: Students run a standard ResNet-50 image classification model. They will execute three tests: one targeted at CPU, one at GPU, and one at NPU.

Measurement: Use the benchmark_app (packaged with OpenVINO) to capture Latency and Throughput.

Key Learning: Students observe Task Manager performance graphs. They should see a ~10-15x performance jump in latency on the NPU compared to the CPU, while the CPU usage remains near 0%.

Command: ```powershell

NPU Throughput Test
benchmark_app -m resnet50.xml -d NPU -hint throughput

## Download a Test Model
We are doing to work with the "Vision Hierarchy" (The 50 Layers) model. It was released in 2015, and is still notable in the history of image classification. It recognizes over 1,000 different types of objects.

The "50" in ResNet-50 stands for the 50 layers of neurons the data must pass through. Think of this like a filter system:
- Initial Layers (The Simple Eye): These layers look for basic things like lines, edges, and dots. They don't know what a "dog" is; they just see a vertical line or a brown curve.
- Middle Layers (The Pattern Finder): These layers combine the lines into shapes. They start recognizing textures (like fur), shapes (like circles), or patterns (like a grid).
- Final Layers (The Concept Builder): These layers take those patterns and say, "I see two triangles (ears) and a wet circle (nose) on a furry texture... this is a Golden Retriever."

The "ResNet" part refers to the Residual Networks family.

Read more at: https://blog.roboflow.com/what-is-resnet-50/

~~~
# Create a folder for our models
mkdir models
cd models

# Download the ResNet-50 model in ONNX format
# (This is a standard 100MB model for image classification)
curl.exe -L https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx --output resnet50.onnx

cd ..
~~~


## Performance Baseline (CPU)
OpenVINO includes the benchmark_app feature.

~~~
benchmark_app --help
benchmark_app -m models/resnet50.onnx -d CPU -t 30
~~~

What to do while it runs:
- Open Task Manager (Ctrl+Shift+Esc)
- Watch the CPU graph; it should spike toward 100% usage

Note the Throughput (FPS) and Latency (ms) reported in the terminal at the end.

## Graphics Assist (GPU)
~~~
benchmark_app -m models/resnet50.onnx -d GPU -t 30
~~~

What to do while it runs:
- Open Task Manager (Ctrl+Shift+Esc)
- Wach the GPU 0 graph
- Notice the CPU graph; it should drop significantly compared to the first test

Check if the throughput (FPS) has increased. Generally, the GPU handles parallel tasks better than the CPU.

## Intel AI Boost (NPU)
~~~
benchmark_app -m models/resnet50.onnx -d NPU -t 30
~~~

What to do while it runs:
- Open Task Manager (Ctrl+Shift+Esc)
- Wach the NPU graph; it will spike to 100%
- Look at the CPU and GPU graphs. They should be flat (near 0%)

Note the throughput (FPS) has increased. The NPU is doing the exact same amount of work as the CPU was, but "silently" without "stealing" resources from the rest of the computer.

## Analysis
| Target Device | FPS | Median Latency (ms) | CPU Load (%) |
|---|---|---|---|
| CPU  |   |   | ~90-100%  |
| GPU  |   |   | ~15-20%  |
| NPU  |   |   | <5%  |
