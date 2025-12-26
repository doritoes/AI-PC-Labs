import cv2
import numpy as np
import openvino as ov
from ultralytics import YOLO
import os
import urllib.request
import time
import shutil
import csv

def setup_and_run_complete_lab():
    # --- 1. DIRECTORY & ASSET SETUP ---
    model_dir = "models"
    yolo_ov_dir = os.path.join(model_dir, "yolov8n_openvino_model")
    face_xml = os.path.join(model_dir, "face_detection.xml")
    face_bin = os.path.join(model_dir, "face_detection.bin")
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Create CSV filename with current date/time
    csv_file = f"security_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    # --- 2. AUTOMATIC MODEL DOWNLOADS & EXPORTS ---
    # Download and Export YOLOv8 to OpenVINO FP16 for NPU
    if not os.path.exists(yolo_ov_dir):
        print("Setup: Downloading/Exporting YOLOv8 for NPU...")
        tmp_model = YOLO('yolov8n.pt')
        export_path = tmp_model.export(format='openvino', half=True) 
        shutil.move(export_path, yolo_ov_dir)

    # Download OpenVINO Face Detection Model
    if not os.path.exists(face_xml):
        print("Setup: Downloading Face Detection IR files...")
        base_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/face-detection-retail-0004/FP16/"
        urllib.request.urlretrieve(base_url + "face-detection-retail-0004.xml", face_xml)
        urllib.request.urlretrieve(base_url + "face-detection-retail-0004.bin", face_bin)

    # --- 3. HARDWARE & LOGGING INITIALIZATION ---
    core = ov.Core()
    target = "NPU" if "NPU" in core.available_devices else "CPU"
    
    # Load models into NPU
    c_face = core.compile_model(core.read_model(face_xml), target)
    c_yolo = core.compile_model(core.read_model(os.path.join(yolo_ov_dir, "yolov8n.xml")), target)
    
    class_names = YOLO('yolov8n.pt').names
    office_items = ['person', 'chair', 'book', 'cup', 'laptop', 'keyboard', 'mouse', 'monitor']
    
    # Initialize CSV with headers
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Event', 'Object_Count', 'Items_Found'])

    cap = cv2.VideoCapture(0)
    last_log_time = 0
    active_items = set()

    print("\n" + "="*50)
    print(f"LAB ACTIVE ON: {target}")
    print(f"LOGGING TO: {csv_file}")
    print(">> TO QUIT: Select the video window and press 'Q'")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]

        # --- TASK A: NPU FACE BLUR (Privacy) ---
        blob_f = cv2.resize(frame, (300, 300)).transpose((2, 0, 1)).reshape(1, 3, 300, 300)
        f_res = c_face([blob_f])[c_face.output(0)]
        for f in f_res[0][0]:
            if f[2] > 0.5:
                x1, y1, x2, y2 = int(f[3]*w), int(f[4]*h), int(f[5]*w), int(f[6]*h)
                roi = frame[max(0,y1):y2, max(0,x1):x2]
                if roi.size > 0:
                    frame[max(0,y1):y2, max(0,x1):x2] = cv2.GaussianBlur(roi, (99, 99), 30)

        # --- TASK B: YOLOv8 DETECTION ---
        # Normalize and resize
        blob_y = cv2.resize(frame, (640, 640)).astype(np.float32) / 255.0
        blob_y = blob_y.transpose((2, 0, 1)).reshape(1, 3, 640, 640)
        
        y_raw = c_yolo([blob_y])[c_yolo.output(0)]
        predictions = np.squeeze(y_raw).T # Transpose to get boxes in rows
        
        boxes, confs, class_ids = [], [], []
        frame_items = set()

        for pred in predictions:
            scores = pred[4:]
            c_id = np.argmax(scores)
            conf = scores[c_id]
            label = class_names[c_id]
            
            # Use sensitive threshold for books (0.35) and stable for others (0.55)
            thresh = 0.35 if label == 'book' else 0.55
            
            if conf > thresh and label in office_items:
                cx, cy, bw, bh = pred[:4]
                rx, ry = int((cx - bw/2) * (w/640)), int((cy - bh/2) * (h/640))
                rw, rh = int(bw * (w/640)), int(bh * (h/640))
                
                boxes.append([rx, ry, rw, rh])
                confs.append(float(conf))
                class_ids.append(c_id)

        # Apply NMS to prevent the "Sea of Boxes"
        indices = cv2.dnn.NMSBoxes(boxes, confs, 0.35, 0.4)
        
        if len(indices) > 0:
            for i in indices.flatten():
                name = class_names[class_ids[i]].upper()
                frame_items.add(name)
                x, y, bw, bh = boxes[i]
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} {confs[i]:.2f}", (x, y-10), 0, 0.5, (0, 255, 0), 2)

        # --- TASK C: CSV EVENT LOGGING ---
        if (frame_items != active_items) and (time.time() - last_log_time > 1.5):
            t_stamp = time.strftime("%H:%M:%S")
            items_str = ", ".join(frame_items) if frame_items else "None"
            
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([t_stamp, "WORKSPACE_CHANGE", len(frame_items), items_str])
            
            print(f"[{t_stamp}] SECURITY EVENT: {items_str}")
            active_items = frame_items
            last_log_time = time.time()

        cv2.imshow("NPU Intelligent Security Lab", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nLab Complete. CSV Data saved to: {csv_file}")

if __name__ == "__main__":
    setup_and_run_complete_lab()
