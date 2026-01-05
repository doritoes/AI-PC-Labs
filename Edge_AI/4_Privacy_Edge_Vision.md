# Privacy-First Edge Vision
In traditional "Cloud AI," your webcam feed is sent over the internet to a server (like Google or Amazon). The server analyzes the video and sends the result back.

"Privacy-First" Edge Vision flips this:
- The Raw Video Never Leaves the PC: The NPU processes the frames locally in the system's memory
- Anonymization at the Source: We will use a Face Detection model to find faces and immediately apply a Gaussian Blur
- GDPR/Privacy Compliance: Since the raw, identifiable faces are blurred before they could ever be saved or uploaded, the system is "private by design"

Goal: Build a "Smart Surveillance" application that detects objects (people, vehicles) in a video stream using only local hardware.

Process: Use a YOLOv8 (You Only Look Once) model. Students will write a Python script using opencv to capture a webcam feed or video file and send frames to the NPU for processing.

Measurement: "Inference time per frame" (ms).

Key Learning: "Double Buffering." Students learn how the CPU handles the "Pre-processing" (resizing images) and "Post-processing" (drawing boxes) while the NPU handles the "Inference" (the actual math) in parallel.

Deliverable: A live video window with real-time bounding boxes running smoothly on the NPU.

## Install ultralytics
1. Ensure the Python virtual environment is active
    - Open a new PowerShell window
    - `cd "$env:USERPROFILE\Edge-AI"`
    - `.\nputest_env\Scripts\Activate`
2. `pip install ultralytics`

## Download the Model
The XML file is the blueprint or structure. The BIN file is the brains, the weighted values
1. Open PowerShell
2. `cd "$env:USERPROFILE\Edge-AI"`
3. `cd models`
4. `curl.exe -L "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/face-detection-retail-0004/FP16/face-detection-retail-0004.xml" --output face_detection.xml`
5. `curl.exe -L "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/face-detection-retail-0004/FP16/face-detection-retail-0004.bin" --output face_detection.bin`
6. `cd ..`


## Run the Smart Privacy Surveillance Script
This script manages two different NPU "inference requests" in a single loop.

1. Plug in your webcam
2. Copy the script [smart_privacy_npu.py](smart_privacy_npu.py)
3. `python smart_privacy_npu.py`
4. Test detection of various office objects (some work better than others)
    - person
    - chair
    - book
    - cup
    - laptop
    - keyboard
    - mouse
    - monitor
5. From the task manager, check the impact on the CPU, GPU, and NPU
6. From the video window press 'q' to quit

## Experiment and Learn
This lab demonstrates a "Metadata Only" strategy. It doesn't just record video.
1. The NPU processes the video
2. The privacy filter blurs the faces
3. The Logger creates a tine text file (.csv) that contains the important info

Compare
- Open the surveillance_log_*.csv files in Excel or Notepad
- How large is the log file after 1 minute of running? (usually only a few KB)
- How large would 1 minute of 4K video be? (hundreds of MB)

Reviewing log data:
- `Import-Csv (Get-ChildItem security_log_*.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 1) | Out-GridView`
- `$log = Import-Csv (Get-ChildItem security_log_*.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 1)`
- `$log | Group-Object Items_Found | Select-Object Name, Count | Sort-Object Count -Descending`

Stress Test
- Walk in and out of the camera's view
- What happens where there are two people? Two chairs?
- Try partially obscuring your face with your hard to defeat the blur
- Hold up a laptop or a chair
- Verify that the CSV file correctly identifies the time and the object
- Try swapping the larger yolov8s.pt in to replace yolov8n.pt model. Do you get better detection?
