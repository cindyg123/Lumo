"""
Test script for Arducam OV2311 USB camera.
Press 'q' to quit.
"""

import cv2

cap = cv2.VideoCapture(8)

if not cap.isOpened():
    cap = cv2.VideoCapture(9)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera open. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Arducam OV2311", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
