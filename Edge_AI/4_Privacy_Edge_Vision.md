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


## Download the Detection Models
We will use the SSDLite MobileNet V2 for general objects and the Face Detection Retail model for the privacy layer.
1. Open PowerShell and start in the project folder
    - `cd "$env:USERPROFILE\Edge-AI"`
2. `cd models`
3. `curl.exe -L https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/ssdlite_mobilenet_v2/FP16/ssdlite_mobilenet_v2.xml --output office_detection.xml`
4. `curl.exe -L https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/ssdlite_mobilenet_v2/FP16/ssdlite_mobilenet_v2.bin --output office_detection.bin`
5. `curl.exe -L https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/face-detection-retail-0004/FP16/face-detection-retail-0004.xml --output face_detection.xml`
6. `curl.exe -L https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/face-detection-retail-0004/FP16/face-detection-retail-0004.bin --output face_detection.bin`
5. `cd ..`

## Create the Smart Privacy Surveillance Script
This script manages two different NPU "inference requests" in a single loop.

smart_privacy_npu.py
~~~
import cv2
import openvino as ov
import numpy as np

def run_combined_lab():
    core = ov.Core()
    
    # Load Models to NPU
    obj_model = core.compile_model(core.read_model("models/office_detection.xml"), "NPU")
    face_model = core.compile_model(core.read_model("models/face_detection.xml"), "NPU")

    # Labels for the Office Model
    target_labels = {1: "PERSON", 15: "CHAIR", 62: "MONITOR", 63: "LAPTOP", 67: "TABLE"}
    
    cap = cv2.VideoCapture(0)
    print("Dual-Model NPU Surveillance Active. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]

        # --- STEP 1: FACE DETECTION & BLUR ---
        face_blob = cv2.resize(frame, (300, 300)).transpose((2, 0, 1)).reshape(1, 3, 300, 300)
        face_results = face_model(face_blob)[face_model.output(0)].reshape(-1, 7)
        
        for face in face_results:
            if face[2] > 0.5: # Confidence
                fx1, fy1 = int(face[3] * w), int(face[4] * h)
                fx2, fy2 = int(face[5] * w), int(face[6] * h)
                # Crop and Blur
                face_zone = frame[max(0,fy1):min(h,fy2), max(0,fx1):min(w,fx2)]
                if face_zone.size > 0:
                    blurred = cv2.GaussianBlur(face_zone, (51, 51), 30)
                    frame[max(0,fy1):min(h,fy2), max(0,fx1):min(w,fx2)] = blurred

        # --- STEP 2: OBJECT DETECTION & BOXES ---
        obj_blob = cv2.resize(frame, (300, 300)).transpose((2, 0, 1)).reshape(1, 3, 300, 300)
        obj_results = obj_model(obj_blob)[obj_model.output(0)].reshape(-1, 7)

        for obj in obj_results:
            conf = obj[2]
            class_id = int(obj[1])
            if conf > 0.5 and class_id in target_labels:
                label = target_labels[class_id]
                ox1, oy1 = int(obj[3] * w), int(obj[4] * h)
                ox2, oy2 = int(obj[5] * w), int(obj[6] * h)
                
                # Draw Box and Label
                cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (ox1, oy1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("NPU Smart Privacy Surveillance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_combined_lab()
~~~

## Run the Lab
The "Handoff": OpenCV (running on CPU) handles the webcam and the blur effect, but OpenVINO (running on NPU) handles the "intelligence" of finding the face.

1. Connect the webcam to the HP Mini
2. `smart_privacy_npu.py`
3. Observe
    * Smoothness Test: Look at the video window. Is it laggy? (Running two AI models on standard CPU usually causes video to stutter; the NPU should be providing nearly 30+ FPS)
    * Task Manager Check: Switch to Performance tab; NPU should show steady "sawtooth" or solid block of usage; CPU should remain very low (5-10%) as it is only "displaying" the window, not doing the AI math

## Adding Logging
smart_surveillance_final.py

~~~
import cv2
import openvino as ov
import numpy as np
import csv
from datetime import datetime
import os

def run_logged_surveillance():
    # --- SETUP LOG FILE ---
    log_file = "surveillance_log.csv"
    # Create header if file doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Object", "Confidence"])

    # --- INITIALIZE NPU ---
    core = ov.Core()
    obj_model = core.compile_model(core.read_model("models/office_detection.xml"), "NPU")
    face_model = core.compile_model(core.read_model("models/face_detection.xml"), "NPU")

    target_labels = {1: "PERSON", 15: "CHAIR", 62: "MONITOR", 63: "LAPTOP", 67: "TABLE"}
    cap = cv2.VideoCapture(0)

    print(f"Surveillance active. Logging events to {log_file}...")

    while True:
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # --- PRIVACY: FACE BLUR ---
        face_blob = cv2.resize(frame, (300, 300)).transpose((2, 0, 1)).reshape(1, 3, 300, 300)
        face_results = face_model(face_blob)[face_model.output(0)].reshape(-1, 7)
        for face in face_results:
            if face[2] > 0.5:
                fx1, fy1, fx2, fy2 = int(face[3]*w), int(face[4]*h), int(face[5]*w), int(face[6]*h)
                face_zone = frame[max(0,fy1):min(h,fy2), max(0,fx1):min(w,fx2)]
                if face_zone.size > 0:
                    frame[max(0,fy1):min(h,fy2), max(0,fx1):min(w,fx2)] = cv2.GaussianBlur(face_zone, (51, 51), 30)

        # --- OBJECT DETECTION & LOGGING ---
        obj_blob = cv2.resize(frame, (300, 300)).transpose((2, 0, 1)).reshape(1, 3, 300, 300)
        obj_results = obj_model(obj_blob)[obj_model.output(0)].reshape(-1, 7)

        detected_this_frame = []
        for obj in obj_results:
            conf = obj[2]
            class_id = int(obj[1])
            if conf > 0.6 and class_id in target_labels:
                label = target_labels[class_id]
                # Log the detection to CSV
                with open(log_file, mode='a', newline='') as f:
                    csv.writer(f).writerow([timestamp, label, round(float(conf), 2)])
                
                # Visual Feedback
                ox1, oy1, ox2, oy2 = int(obj[3]*w), int(obj[4]*h), int(obj[5]*w), int(obj[6]*h)
                cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}", (ox1, oy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Smart NPU Surveillance (Logged)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_logged_surveillance()
~~~

## Experiment and Learn
This lab demonstrates a "Metadata Only" strategy. It doesn't just record video.
1. The NPU processes the video
2. The privacy filter blurs the faces
3. The Logger creates a tine text file (.csv) that contains the important info
Explain to students that a professional surveillance system doesn't just record video. Instead:

Compare
- Open the surveillance_log.csv in Excel or Notepad
- How large is the log file after 1 minute of running? (usually only a few KB)
- How large would 1 minute of 4K video be? (hundreds of MB)

Stress Test
- Walk in and out of the camera's view
- Hold up a laptop or a chair
- Verify that the CSV file correctly identifies the time and the object
