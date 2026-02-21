"""
NIR Camera Face Heatmap
Captures frames from the Arducam OV2311 NIR camera and
applies a heatmap colormap for a thermal-style visualization.
Press 'q' to quit.
"""
import cv2

# Open USB camera (Arducam OV2311)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(8)
if not cap.isOpened():
    cap = cv2.VideoCapture(9)
if not cap.isOpened():
    print("Error: Could not open USB camera!")
    exit(1)

print("NIR Heatmap running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Convert to grayscale (NIR camera may already be grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply heatmap colormap
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Display both side by side
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    combined = cv2.hconcat([gray_bgr, heatmap])
    combined = cv2.resize(combined, (1200, 450))
    cv2.imshow("NIR Camera - Original | Heatmap", combined)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
