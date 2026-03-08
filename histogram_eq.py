"""
Histogram Equalization - RGB Camera (IMX519)
Shows live comparison: Original vs Standard HE vs CLAHE
Press 'q' to quit.
"""
import cv2
import numpy as np
import time
from picamera2 import Picamera2

# Initialize RGB Camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1920, 1080)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(1.0)
picam2.set_controls({"AfMode": 2})

# CLAHE settings
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

print("Histogram Equalization running. Press 'q' to quit.")

while True:
    frame = picam2.capture_array()
    frame = cv2.flip(frame, 1)

    # Convert to different color spaces for processing
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # --- Standard Histogram Equalization ---
    yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    he_result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # --- CLAHE (Adaptive) ---
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    clahe_result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Add labels
    cv2.putText(frame_bgr, "Original", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(he_result, "Histogram EQ", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(clahe_result, "CLAHE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Resize for side-by-side display
    h, w = frame_bgr.shape[:2]
    small_w = w // 3
    small_h = h // 3
    orig_small = cv2.resize(frame_bgr, (small_w, small_h))
    he_small = cv2.resize(he_result, (small_w, small_h))
    clahe_small = cv2.resize(clahe_result, (small_w, small_h))

    combined = cv2.hconcat([orig_small, he_small, clahe_small])
    cv2.imshow("Lumo - Histogram Equalization Comparison", combined)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q') or key == 27:
        break

print("Done!")
picam2.close()
cv2.destroyAllWindows()
