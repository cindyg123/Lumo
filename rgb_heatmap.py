"""
RGB Camera Heatmap + Drowsiness Detection
Combines heatmap visualization with face mesh, EAR eye tracking, PERCLOS, blink counter,
60-second timer, heat ratios, and voice results.
Left side: RGB camera with face mesh + detection overlays
Right side: Heatmap with heat ratios
Press 'q' to quit, 'r' to reset head pose.
"""
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import pyttsx3
from collections import deque
from picamera2 import Picamera2

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def rotation_matrix_to_euler_angles(R):
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z], dtype=np.float64)

def euclidean_dist(pt1, pt2):
    return np.linalg.norm(pt1 - pt2)

def calculate_ear(eye_top, eye_bottom, eye_left, eye_right):
    top = np.array([eye_top.x, eye_top.y])
    bottom = np.array([eye_bottom.x, eye_bottom.y])
    left = np.array([eye_left.x, eye_left.y])
    right = np.array([eye_right.x, eye_right.y])
    vertical_dist = euclidean_dist(top, bottom)
    horizontal_dist = euclidean_dist(left, right)
    return vertical_dist / (horizontal_dist + 1e-6)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize Raspberry Pi Camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1920, 1080)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(1.0)
picam2.set_controls({"AfMode": 2})

# 3D reference points for head pose estimation
reference_3d_points = np.array([
    [0.0, 40.0, 0.0],
    [0.0, -50.0, 0.0],
    [-40.0, 0.0, -30.0],
    [40.0, 0.0, -30.0],
    [0.0, 0.0, 50.0],
    [-25.0, -40.0, -25.0],
    [25.0, -40.0, -25.0],
], dtype=np.float64)
landmark_ids_pose = [10, 152, 234, 454, 1, 78, 308]

# Drowsiness detection settings
ear_threshold = 0.35
eye_closed_frames_threshold = 15
left_eye_closed_count = 0
right_eye_closed_count = 0

# PERCLOS: percentage of eye closure over a rolling window
perclos_window = deque(maxlen=150)
perclos_value = 0.0
perclos_threshold = 40.0

# Head pose calibration offsets
pitch_offset = 0.0
yaw_offset = 0.0
roll_offset = 0.0

# Timer: 60 seconds then auto-quit
timer_duration = 60
timer_start = time.time()

# Blink counter
blink_count = 0
was_blinking = False

print("RGB Heatmap + Detection running. 60 second timer started!")

while True:
    frame = picam2.capture_array()
    start_time = time.time()

    frame = cv2.flip(frame, 1)
    rgb_frame = frame.copy()

    # Process face landmarks
    results = face_mesh.process(rgb_frame)
    img_h, img_w, _ = frame.shape

    eye_status_text = ""
    display_pitch = 0
    display_yaw = 0
    display_roll = 0

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Head pose estimation
        face_2d_points = []
        for idx in landmark_ids_pose:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d_points.append([x, y])
        face_2d_points = np.array(face_2d_points, dtype=np.float64)

        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success_pnp, rotation_vector, translation_vector = cv2.solvePnP(
            reference_3d_points, face_2d_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success_pnp:
            rmat, _ = cv2.Rodrigues(rotation_vector)
            euler_angles = rotation_matrix_to_euler_angles(rmat) * (180.0 / math.pi)
            pitch, yaw, roll = euler_angles
            display_pitch = pitch - pitch_offset
            display_yaw = yaw - yaw_offset
            display_roll = roll - roll_offset

            # Draw nose direction line
            nose_idx = landmark_ids_pose.index(1)
            nose_2d = face_2d_points[nose_idx]
            nose_end_3d = np.array([[0, 0, 100]], dtype=np.float64)
            nose_end_2d, _ = cv2.projectPoints(
                nose_end_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs
            )
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_end_2d[0][0][0]), int(nose_end_2d[0][0][1]))
            cv2.line(frame, p1, p2, (255, 0, 0), 2)

        # Eye closure detection
        l_top = face_landmarks.landmark[159]
        l_bottom = face_landmarks.landmark[145]
        l_left = face_landmarks.landmark[33]
        l_right = face_landmarks.landmark[133]
        r_top = face_landmarks.landmark[386]
        r_bottom = face_landmarks.landmark[374]
        r_left = face_landmarks.landmark[263]
        r_right = face_landmarks.landmark[362]

        left_ear = calculate_ear(l_top, l_bottom, l_left, l_right)
        right_ear = calculate_ear(r_top, r_bottom, r_left, r_right)

        if left_ear < ear_threshold:
            left_eye_closed_count += 1
        else:
            left_eye_closed_count = 0
        if right_ear < ear_threshold:
            right_eye_closed_count += 1
        else:
            right_eye_closed_count = 0

        # PERCLOS tracking
        both_closed = (left_ear < ear_threshold and right_ear < ear_threshold)
        perclos_window.append(1 if both_closed else 0)
        if len(perclos_window) > 0:
            perclos_value = (sum(perclos_window) / len(perclos_window)) * 100

        # Blink counting
        eyes_closed_now = (left_ear < ear_threshold and right_ear < ear_threshold)
        if was_blinking and not eyes_closed_now:
            blink_count += 1
        was_blinking = eyes_closed_now

        # Determine eye status
        if (left_eye_closed_count > eye_closed_frames_threshold and
            right_eye_closed_count > eye_closed_frames_threshold):
            eye_status_text = "SLEEPING!"
        elif (left_eye_closed_count > 0 or right_eye_closed_count > 0):
            eye_status_text = "Blinking"
        else:
            eye_status_text = "Eyes Open"

        # Draw face mesh
        mp_drawing.draw_landmarks(
            frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec
        )

    # Calculate FPS
    end_time = time.time()
    fps = 1.0 / (end_time - start_time + 1e-6)

    # Generate face-only heatmap BEFORE drawing text
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Start with a black background for heatmap
    heatmap = np.zeros_like(frame)

    if results.multi_face_landmarks:
        # Get face bounding box from landmarks
        face_lms = results.multi_face_landmarks[0]
        xs = [int(lm.x * img_w) for lm in face_lms.landmark]
        ys = [int(lm.y * img_h) for lm in face_lms.landmark]
        x_min = max(0, min(xs) - 20)
        x_max = min(img_w, max(xs) + 20)
        y_min = max(0, min(ys) - 20)
        y_max = min(img_h, max(ys) + 20)

        # Apply heatmap only to face region
        face_gray = gray[y_min:y_max, x_min:x_max]
        face_heatmap = cv2.applyColorMap(face_gray, cv2.COLORMAP_JET)
        heatmap[y_min:y_max, x_min:x_max] = face_heatmap
    else:
        # No face detected, show full heatmap as fallback
        heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Draw text overlays on camera side only
    if eye_status_text == "SLEEPING!":
        eye_color = (0, 0, 255)
    elif eye_status_text == "Blinking":
        eye_color = (0, 255, 255)
    else:
        eye_color = (0, 255, 0)
    cv2.putText(frame, f"Eyes: {eye_status_text}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, eye_color, 4)

    perclos_color = (0, 0, 255) if perclos_value > perclos_threshold else (0, 255, 0)
    cv2.putText(frame, f"PERCLOS: {perclos_value:.1f}%", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 2.0, perclos_color, 4)
    cv2.putText(frame, f"Blinks: {blink_count}", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)

    # Timer countdown
    elapsed = time.time() - timer_start
    remaining = max(0, timer_duration - elapsed)
    minutes = int(remaining) // 60
    seconds = int(remaining) % 60
    timer_color = (0, 0, 255) if remaining < 10 else (0, 255, 255)
    cv2.putText(frame, f"Timer: {minutes}:{seconds:02d}", (40, 320), cv2.FONT_HERSHEY_SIMPLEX, 2.0, timer_color, 4)

    cv2.putText(frame, f"FPS: {int(fps)}", (40, img_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

    # Auto-quit when timer runs out
    if remaining <= 0:
        print(f"\nTime's up! Total blinks in 60 seconds: {blink_count}")
        print(f"PERCLOS: {perclos_value:.1f}%")
        print(f"Hot: {hot_ratio:.1f}%, Warm: {warm_ratio:.1f}%, Cool: {cool_ratio:.1f}%")
        # Say the results out loud with pauses
        engine.say(f"{blink_count} blinks in 60 seconds.")
        engine.runAndWait()
        time.sleep(1.5)
        engine.say(f"{hot_ratio:.0f} percent hot.")
        engine.runAndWait()
        time.sleep(1.5)
        engine.say(f"{warm_ratio:.0f} percent warm.")
        engine.runAndWait()
        time.sleep(1.5)
        engine.say(f"{cool_ratio:.0f} percent cool.")
        engine.runAndWait()
        break

    # Calculate heat ratios (face region only if detected)
    if results.multi_face_landmarks:
        face_region = gray[y_min:y_max, x_min:x_max]
    else:
        face_region = gray
    total_pixels = face_region.shape[0] * face_region.shape[1]
    hot_pixels = np.count_nonzero(face_region > 200)
    warm_pixels = np.count_nonzero(face_region > 150)
    cool_pixels = total_pixels - warm_pixels

    hot_ratio = (hot_pixels / total_pixels) * 100
    warm_ratio = (warm_pixels / total_pixels) * 100
    cool_ratio = (cool_pixels / total_pixels) * 100

    # Display heat ratios on heatmap side
    cv2.putText(heatmap, f"Hot (red): {hot_ratio:.1f}%", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
    cv2.putText(heatmap, f"Warm (yellow): {warm_ratio:.1f}%", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
    cv2.putText(heatmap, f"Cool (blue): {cool_ratio:.1f}%", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)

    # Side by side: RGB with overlays | Heatmap with ratios
    combined = cv2.hconcat([frame, heatmap])
    combined = cv2.resize(combined, (1400, 500))
    cv2.imshow("Lumo - RGB + Heatmap + Detection", combined)

    key = cv2.waitKey(5) & 0xFF
    if key in [27, ord('q')]:
        break
    elif key == ord('r'):
        pitch_offset = display_pitch
        yaw_offset = display_yaw
        roll_offset = display_roll
        print("Pose reset to 0.")

picam2.close()
cv2.destroyAllWindows()
