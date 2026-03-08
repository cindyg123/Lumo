"""
Face Extraction - RGB Camera (IMX519)
Detects faces using MediaPipe, draws bounding boxes, and saves cropped face images.
Press 's' to save current face, 'q' to quit.
Auto-saves a face every 5 seconds if detected.
"""
import cv2
import mediapipe as mp
import numpy as np
import os
import time
from picamera2 import Picamera2

# Create output folder
output_dir = "/home/wrecker888/Documents/SafeSense/faces"
os.makedirs(output_dir, exist_ok=True)

# Initialize MediaPipe Face Detection (faster than Face Mesh for extraction)
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.4
)

# Initialize RGB Camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1920, 1080)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(1.0)
picam2.set_controls({"AfMode": 2})

save_count = 0
last_auto_save = time.time()
auto_save_interval = 5  # seconds between auto-saves

print(f"Face Extraction running. Saving faces to {output_dir}/")
print("Press 's' to save face manually, 'q' to quit.")

while True:
    frame = picam2.capture_array()
    frame = cv2.flip(frame, 1)
    img_h, img_w, _ = frame.shape

    # Detect faces
    results = face_detection.process(frame)

    face_crop = None

    if results.detections:
        for detection in results.detections:
            # Get bounding box
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * img_w)
            y = int(bbox.ymin * img_h)
            w = int(bbox.width * img_w)
            h = int(bbox.height * img_h)

            # Add padding around face
            pad = 40
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img_w, x + w + pad)
            y2 = min(img_h, y + h + pad)

            # Crop face
            face_crop = frame[y1:y2, x1:x2].copy()

            # Draw bounding box on display
            confidence = detection.score[0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"Face: {confidence:.0%}", (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            # Auto-save every interval
            if time.time() - last_auto_save > auto_save_interval:
                save_count += 1
                filename = f"face_{save_count:04d}.jpg"
                filepath = os.path.join(output_dir, filename)
                face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, face_bgr)
                print(f"Auto-saved: {filename} ({face_crop.shape[1]}x{face_crop.shape[0]})")
                last_auto_save = time.time()
    else:
        cv2.putText(frame, "No face detected", (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

    # Show count
    cv2.putText(frame, f"Saved: {save_count}", (40, img_h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)

    # Display
    display = cv2.resize(frame, (960, 540))
    cv2.imshow("Lumo - Face Extraction", display)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('s') and face_crop is not None:
        save_count += 1
        filename = f"face_{save_count:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
        face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, face_bgr)
        print(f"Manual save: {filename} ({face_crop.shape[1]}x{face_crop.shape[0]})")
        last_auto_save = time.time()

print(f"\nDone! Saved {save_count} face images to {output_dir}/")
picam2.close()
cv2.destroyAllWindows()
