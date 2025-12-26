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


## Create the Smart Privacy Surveillance Script
This script manages two different NPU "inference requests" in a single loop.

smart_privacy_npu.py
~~~
import cv2
import numpy as np
import openvino as ov
from ultralytics import YOLO
import os
import time

def run_context_aware_lab():
    # 1. HARDWARE INIT
    model_path = "models/yolov8n_openvino_model/yolov8n.xml"
    face_path = "models/face_detection.xml"
    
    core = ov.Core()
    device = "NPU" if "NPU" in core.available_devices else "CPU"
    
    c_yolo = core.compile_model(core.read_model(model_path), device)
    c_face = core.compile_model(core.read_model(face_path), device)
    
    class_names = YOLO('yolov8n.pt').names
    # Expanded list to prevent "Mis-labeling" (e.g., keyboard -> laptop)
    office_list = ['person', 'chair', 'book', 'cup', 'laptop', 'keyboard', 'mouse', 'monitor', 'tv']
    
    cap = cv2.VideoCapture(0)

    print("\n" + "="*50)
    print("OFFICE CONTEXT LAB: ACTIVE")
    print(">> QUIT: Press 'Q' in the video window.")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]

        # --- TASK A: NPU FACE BLUR ---
        f_blob = cv2.resize(frame, (300, 300)).transpose((2, 0, 1)).reshape(1, 3, 300, 300)
        f_hits = c_face([f_blob])[c_face.output(0)]
        for hit in f_hits[0][0]:
            if hit[2] > 0.5:
                x1, y1, x2, y2 = int(hit[3]*w), int(hit[4]*h), int(hit[5]*w), int(hit[6]*h)
                roi = frame[max(0,y1):y2, max(0,x1):x2]
                if roi.size > 0:
                    frame[max(0,y1):y2, max(0,x1):x2] = cv2.GaussianBlur(roi, (99, 99), 30)

        # --- TASK B: TUNED OBJECT DETECTION ---
        input_img = cv2.resize(frame, (640, 640)).astype(np.float32) / 255.0
        blob = input_img.transpose(2, 0, 1).reshape(1, 3, 640, 640)
        
        results = c_yolo([blob])[c_yolo.output(0)]
        data = np.squeeze(results).T 
        
        boxes, confs, ids = [], [], []
        for row in data:
            scores = row[4:]
            c_id = np.argmax(scores)
            conf = scores[c_id]
            
            # DYNAMIC THRESHOLD: Lower threshold for items, keep it high for person
            # This helps the NPU 'see' the book and cup which are smaller/lower contrast
            thresh = 0.45 if class_names[c_id] in ['book', 'cup', 'chair'] else 0.60
            
            if conf > thresh and class_names[c_id] in office_list:
                cx, cy, bw, bh = row[:4]
                rx, ry = int((cx - bw/2) * (w/640)), int((cy - bh/2) * (h/640))
                rw, rh = int(bw * (w/640)), int(bh * (h/640))
                
                boxes.append([rx, ry, rw, rh])
                confs.append(float(conf))
                ids.append(c_id)

        # NMS cleanup to prevent box flickering
        indices = cv2.dnn.NMSBoxes(boxes, confs, 0.45, 0.4)
        
        if len(indices) > 0:
            for i in indices.flatten():
                label = class_names[ids[i]].upper()
                bx, by, bw, bh = boxes[i]
                
                # Visuals
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confs[i]:.2f}", (bx, by - 10), 0, 0.5, (0, 255, 0), 2)

        cv2.imshow("NPU Office Context", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_context_aware_lab()
~~~

## Run the Lab
The "Handoff": OpenCV (running on CPU) handles the webcam and the blur effect, but OpenVINO (running on NPU) handles the "intelligence" of finding the face.

1. Connect the webcam to the HP Mini
2. `python smart_privacy_npu.py`
    - The first time there is a long delay before the webcam window appears
    - Downloading yoyo,
    - NPU is compiling the model's math into blob format; CPU will spike while preparing the model, then NPU takes over
4. Observe
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
