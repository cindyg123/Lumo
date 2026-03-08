"""
Frame Extraction - RGB Camera (IMX519)
Step 1: Captures raw frames from the camera at regular intervals.
Saves full frames and displays live feed with frame count and FPS.
Press 's' to save manually, 'q' to quit.
Auto-saves a frame every 2 seconds.
"""
import cv2
import numpy as np
import os
import time
from picamera2 import Picamera2

# Create output folder
output_dir = "/home/wrecker888/Documents/SafeSense/frames"
os.makedirs(output_dir, exist_ok=True)

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
auto_save_interval = 2  # seconds between auto-saves
start_time = time.time()

print(f"Frame Extraction running. Saving frames to {output_dir}/")
print("Press 's' to save manually, 'q' to quit.")

while True:
    frame = picam2.capture_array()
    frame = cv2.flip(frame, 1)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Calculate FPS
    elapsed = time.time() - start_time
    fps_text = f"FPS: {save_count / max(elapsed, 1):.1f} saved/sec"

    # Auto-save every interval
    if time.time() - last_auto_save > auto_save_interval:
        save_count += 1
        filename = f"frame_{save_count:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame_bgr)
        print(f"Saved: {filename} ({frame_bgr.shape[1]}x{frame_bgr.shape[0]})")
        last_auto_save = time.time()

    # Draw overlays on display copy
    display = frame_bgr.copy()
    cv2.putText(display, f"Frames saved: {save_count}", (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
    cv2.putText(display, f"Resolution: {frame_bgr.shape[1]}x{frame_bgr.shape[0]}", (40, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
    cv2.putText(display, f"Time: {int(elapsed)}s", (40, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)

    # Show display
    display = cv2.resize(display, (960, 540))
    cv2.imshow("Lumo - Frame Extraction", display)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('s'):
        save_count += 1
        filename = f"frame_{save_count:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame_bgr)
        print(f"Manual save: {filename}")
        last_auto_save = time.time()

print(f"\nDone! Saved {save_count} frames to {output_dir}/")
picam2.close()
cv2.destroyAllWindows()
