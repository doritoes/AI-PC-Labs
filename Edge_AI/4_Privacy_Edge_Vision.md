# Privacy-First Edge Vision
Privacy-First Edge Vision (Object Detection)

Goal: Build a "Smart Surveillance" application that detects objects (people, vehicles) in a video stream using only local hardware.

Process: Use a YOLOv8 (You Only Look Once) model. Students will write a Python script using opencv to capture a webcam feed or video file and send frames to the NPU for processing.

Measurement: "Inference time per frame" (ms).

Key Learning: "Double Buffering." Students learn how the CPU handles the "Pre-processing" (resizing images) and "Post-processing" (drawing boxes) while the NPU handles the "Inference" (the actual math) in parallel.

Deliverable: A live video window with real-time bounding boxes running smoothly on the NPU.
