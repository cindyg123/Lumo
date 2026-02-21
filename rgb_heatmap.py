"""
RGB Camera Face Heatmap
Captures frames from the Raspberry Pi camera and
applies a heatmap colormap for a thermal-style visualization.
Press 'q' to quit.
"""
import cv2
import numpy as np
import time
from picamera2 import Picamera2

# Initialize Raspberry Pi Camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1920, 1080)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(1.0)
picam2.set_controls({"AfMode": 2})

print("RGB Heatmap running. Press 'q' to quit.")

while True:
    frame = picam2.capture_array()

    # Convert to grayscale for heatmap
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Apply heatmap colormap
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Display normal camera and heatmap side by side
    combined = cv2.hconcat([frame, heatmap])
    combined = cv2.resize(combined, (1200, 450))
    cv2.imshow("RGB Camera - Original | Heatmap", combined)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

picam2.close()
cv2.destroyAllWindows()
