"""
SafeSense - Drowsiness Detection System
This program monitors a person's face to detect signs of drowsiness by:
1. Tracking head position (pitch, yaw, roll angles)
2. Monitoring eye closure using Eye Aspect Ratio (EAR)
3. Triggering audio alerts when drowsiness is detected
"""

# Import required libraries
import cv2  # OpenCV for image processing and display
import mediapipe as mp  # Google's MediaPipe for face landmark detection
import numpy as np  # NumPy for mathematical operations
import math  # Math library for trigonometric calculations
import time  # Time library for FPS calculation
import pyttsx3  # Text-to-speech engine for audio alerts
import os  # Operating system interface for file operations
import json  # JSON library for reading configuration files
from picamera2 import Picamera2  # Raspberry Pi camera interface

# Initialize text-to-speech engine for audio alerts
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set speech rate (words per minute)

# File paths for configuration
ALERT_STYLE_FILE = "/home/wrecker888/Downloads/alert_style.json"  # User's custom alert preferences
DEFAULT_SONG_PATH = "/home/wrecker888/Downloads/annoying_song.mp3"  # Optional alarm song

def play_alert():
    """
    Play an audio alert when drowsiness is detected.
    The alert type depends on the user's configuration in alert_style.json:
    - default: Simple TTS message with user's name
    - tts: Custom TTS message
    - song: Play an annoying song to wake them up
    """
    print(f"[DEBUG] play_alert called")

    # Default alert configuration
    style = {"type": "default", "message": "", "name": "Cindy"}

    # Try to load custom alert settings from JSON file
    if os.path.exists(ALERT_STYLE_FILE):
        print(f"[DEBUG] Found ALERT_STYLE_FILE at: {ALERT_STYLE_FILE}")
        with open(ALERT_STYLE_FILE, "r") as f:
            try:
                config = json.load(f)
                if isinstance(config, dict) and config:
                    # Get the first user's alert settings from the config
                    username, style = next(iter(config.items()))
                    print(f"[DEBUG] Using alert style for user '{username}': {style}")
            except Exception as e:
                print(f"[ERROR] Failed to parse ALERT_STYLE_FILE: {e}")

    # Get the user's name (defaults to "Cindy" if not found)
    name = style.get("name", "Cindy")

    # Play the appropriate alert based on the alert type
    if style["type"] == "tts":
        # Custom text-to-speech message
        msg = style.get("message", f"{name}, wake up!")
        engine.say(msg)
    elif style["type"] == "song":
        # Play an annoying song if the file exists
        if os.path.exists(DEFAULT_SONG_PATH):
            from playsound import playsound
            playsound(DEFAULT_SONG_PATH)
            return
        else:
            engine.say("Annoying song file not found.")
    else:
        # Default alert: simple wake-up message
        engine.say(f"{name}, wake up!")

    # Actually speak the message
    engine.runAndWait()

def rotation_matrix_to_euler_angles(R):
    """
    Convert a 3x3 rotation matrix to Euler angles (pitch, yaw, roll).

    Euler angles describe the orientation of the head:
    - Pitch: Rotation around X-axis (nodding up/down)
    - Yaw: Rotation around Y-axis (turning left/right)
    - Roll: Rotation around Z-axis (tilting head sideways)

    Args:
        R: 3x3 rotation matrix from cv2.Rodrigues

    Returns:
        NumPy array containing [pitch, yaw, roll] in radians
    """
    # Calculate the magnitude of rotation
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)

    # Check if we're in a singularity (gimbal lock) situation
    singular = sy < 1e-6

    if not singular:
        # Normal case: calculate all three angles
        x = math.atan2(R[2, 1], R[2, 2])  # Pitch
        y = math.atan2(-R[2, 0], sy)       # Yaw
        z = math.atan2(R[1, 0], R[0, 0])   # Roll
    else:
        # Singularity case: use alternative calculation
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z], dtype=np.float64)

def euclidean_dist(pt1, pt2):
    """
    Calculate the Euclidean (straight-line) distance between two points.

    Args:
        pt1: First point as NumPy array [x, y]
        pt2: Second point as NumPy array [x, y]

    Returns:
        Distance between the two points
    """
    return np.linalg.norm(pt1 - pt2)

def calculate_ear(eye_top, eye_bottom, eye_left, eye_right):
    """
    Calculate the Eye Aspect Ratio (EAR) to determine if an eye is open or closed.

    EAR is the ratio of vertical to horizontal eye distances:
    - When eye is open: EAR is higher (eye is tall relative to width)
    - When eye is closed: EAR is lower (eye height approaches zero)

    Args:
        eye_top: Top eyelid landmark
        eye_bottom: Bottom eyelid landmark
        eye_left: Left corner of eye landmark
        eye_right: Right corner of eye landmark

    Returns:
        Eye Aspect Ratio (float)
    """
    # Convert landmarks to 2D coordinate arrays
    top = np.array([eye_top.x, eye_top.y])
    bottom = np.array([eye_bottom.x, eye_bottom.y])
    left = np.array([eye_left.x, eye_left.y])
    right = np.array([eye_right.x, eye_right.y])

    # Calculate distances
    vertical_dist = euclidean_dist(top, bottom)      # Height of eye
    horizontal_dist = euclidean_dist(left, right)    # Width of eye

    # Return the ratio (add tiny value to prevent division by zero)
    return vertical_dist / (horizontal_dist + 1e-6)

def main():
    """
    Main function that runs the drowsiness detection system.
    This function:
    1. Sets up the camera and face detection
    2. Continuously processes video frames
    3. Detects head pose and eye closure
    4. Triggers alerts when drowsiness is detected
    """

    # ==================== SETUP PHASE ====================

    # Initialize MediaPipe Face Mesh for detecting facial landmarks
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # Create Face Mesh detector with these settings:
    # - static_image_mode=False: Optimized for video (not static images)
    # - refine_landmarks=True: Get more detailed landmarks around eyes and lips
    # - max_num_faces=1: Only track one face (the driver)
    # - min_detection_confidence=0.5: 50% confidence threshold for initial detection
    # - min_tracking_confidence=0.5: 50% confidence threshold for continued tracking
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize Raspberry Pi Camera
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (1920, 1080)  # Full HD resolution
    picam2.preview_configuration.main.format = "RGB888"     # RGB color format
    picam2.configure("preview")
    picam2.start()
    time.sleep(1.0)  # Wait 1 second for camera to warm up
    picam2.set_controls({"AfMode": 2})  # Enable continuous autofocus

    # 3D reference points for a generic human face model (in millimeters)
    # These correspond to specific facial landmarks and help calculate head pose
    reference_3d_points = np.array([
        [0.0, 40.0, 0.0],        # Forehead
        [0.0, -50.0, 0.0],       # Chin
        [-40.0, 0.0, -30.0],     # Left eye corner
        [40.0, 0.0, -30.0],      # Right eye corner
        [0.0, 0.0, 50.0],        # Nose tip
        [-25.0, -40.0, -25.0],   # Left mouth corner
        [25.0, -40.0, -25.0],    # Right mouth corner
    ], dtype=np.float64)

    # MediaPipe landmark IDs that correspond to the 3D reference points above
    landmark_ids_pose = [10, 152, 234, 454, 1, 78, 308]

    # ==================== DROWSINESS DETECTION SETTINGS ====================

    # Eye Aspect Ratio (EAR) threshold: below this value means eye is closed
    ear_threshold = 0.35

    # Number of consecutive frames eyes must be closed to trigger "sleeping" alert
    # At ~20 FPS, 15 frames = about 0.75 seconds
    eye_closed_frames_threshold = 15

    # Counters to track how many frames each eye has been closed
    left_eye_closed_count = 0
    right_eye_closed_count = 0

    # ==================== HEAD POSE CALIBRATION ====================

    # Offset values to calibrate head pose to "neutral" position
    # User can press 'r' key to reset these to current position
    pitch_offset = 0.0  # Up/down tilt offset
    yaw_offset = 0.0    # Left/right turn offset
    roll_offset = 0.0   # Side tilt offset

    # ==================== MAIN DETECTION LOOP ====================

    while True:
        # Capture a frame from the camera
        frame = picam2.capture_array()
        start_time = time.time()  # For FPS calculation

        # Flip frame horizontally (mirror mode for natural viewing)
        frame = cv2.flip(frame, 1)

        # Convert color from RGB to BGR (MediaPipe expects BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Process the frame to detect facial landmarks
        results = face_mesh.process(rgb_frame)

        # Get frame dimensions
        img_h, img_w, _ = frame.shape

        # Initialize text strings for display
        pitch_text = ""
        yaw_text = ""
        roll_text = ""
        orientation_text = ""
        eye_status_text = ""

        # ==================== FACE DETECTION ====================

        # Check if a face was detected in the frame
        if results.multi_face_landmarks:
            # Get the first (and only) face's landmarks
            face_landmarks = results.multi_face_landmarks[0]

            # ==================== HEAD POSE ESTIMATION ====================

            # Extract the 2D positions of specific landmarks for pose estimation
            face_2d_points = []
            for idx in landmark_ids_pose:
                lm = face_landmarks.landmark[idx]
                # Convert normalized coordinates (0-1) to pixel coordinates
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d_points.append([x, y])
            face_2d_points = np.array(face_2d_points, dtype=np.float64)

            # Create camera matrix (simplified pinhole camera model)
            # This describes the camera's intrinsic properties
            focal_length = img_w  # Approximate focal length
            center = (img_w / 2, img_h / 2)  # Image center point
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

            # Assume no lens distortion for simplicity
            dist_coeffs = np.zeros((4, 1))

            # Solve PnP (Perspective-n-Point) problem to find head pose
            # This calculates rotation and translation of the head relative to camera
            success_pnp, rotation_vector, translation_vector = cv2.solvePnP(
                reference_3d_points,  # 3D model points
                face_2d_points,       # 2D image points
                camera_matrix,        # Camera intrinsics
                dist_coeffs,          # Distortion coefficients
                flags=cv2.SOLVEPNP_ITERATIVE  # Use iterative method for accuracy
            )

            if success_pnp:
                # Convert rotation vector to rotation matrix
                rmat, _ = cv2.Rodrigues(rotation_vector)

                # Convert rotation matrix to Euler angles and convert to degrees
                euler_angles = rotation_matrix_to_euler_angles(rmat) * (180.0 / math.pi)
                pitch, yaw, roll = euler_angles

                # Apply calibration offsets (user can reset with 'r' key)
                display_pitch = pitch - pitch_offset
                display_yaw = yaw - yaw_offset
                display_roll = roll - roll_offset

                # Format text for display
                pitch_text = f"Pitch: {display_pitch:.2f}"
                yaw_text = f"Yaw:   {display_yaw:.2f}"
                roll_text = f"Roll:  {display_roll:.2f}"

                # Determine head orientation based on angles
                pitch_up = (display_pitch > 180 and display_pitch < 353)    # Looking up
                pitch_down = (display_pitch < 180 and display_pitch > 7)    # Looking down
                yaw_left = (display_yaw > 7)                                 # Looking left
                yaw_right = (display_yaw < -7)                               # Looking right

                # Set orientation text based on detected direction
                orientation_text = "Forward"
                if pitch_up:
                    if yaw_left:
                        orientation_text = "Looking Up-Left"
                    elif yaw_right:
                        orientation_text = "Looking Up-Right"
                    else:
                        orientation_text = "Looking Up"
                elif pitch_down:
                    if yaw_left:
                        orientation_text = "Looking Down-Left"
                    elif yaw_right:
                        orientation_text = "Looking Down-Right"
                    else:
                        orientation_text = "Looking Down"
                else:
                    if yaw_left:
                        orientation_text = "Looking Left"
                    elif yaw_right:
                        orientation_text = "Looking Right"

                # Draw a line from nose tip showing head direction (visualization)
                nose_idx = landmark_ids_pose.index(1)  # Nose tip is at index 1
                nose_2d = face_2d_points[nose_idx]

                # Project a 3D point in front of the nose to 2D
                nose_end_3d = np.array([[0, 0, 100]], dtype=np.float64)
                nose_end_2d, _ = cv2.projectPoints(
                    nose_end_3d, rotation_vector, translation_vector,
                    camera_matrix, dist_coeffs
                )

                # Draw the direction line in blue
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_end_2d[0][0][0]), int(nose_end_2d[0][0][1]))
                cv2.line(frame, p1, p2, (255, 0, 0), 2)

            # ==================== EYE CLOSURE DETECTION ====================

            # Get landmarks for left eye (top, bottom, left corner, right corner)
            l_top = face_landmarks.landmark[159]
            l_bottom = face_landmarks.landmark[145]
            l_left = face_landmarks.landmark[33]
            l_right = face_landmarks.landmark[133]

            # Get landmarks for right eye
            r_top = face_landmarks.landmark[386]
            r_bottom = face_landmarks.landmark[374]
            r_left = face_landmarks.landmark[263]
            r_right = face_landmarks.landmark[362]

            # Calculate Eye Aspect Ratio (EAR) for both eyes
            left_ear = calculate_ear(l_top, l_bottom, l_left, l_right)
            right_ear = calculate_ear(r_top, r_bottom, r_left, r_right)

            # Track left eye closure
            if left_ear < ear_threshold:
                left_eye_closed_count += 1  # Eye is closed, increment counter
            else:
                left_eye_closed_count = 0   # Eye is open, reset counter

            # Track right eye closure
            if right_ear < ear_threshold:
                right_eye_closed_count += 1  # Eye is closed, increment counter
            else:
                right_eye_closed_count = 0   # Eye is open, reset counter

            # ==================== DROWSINESS ALERT ====================

            # Check if both eyes have been closed for too long
            if (left_eye_closed_count > eye_closed_frames_threshold and
                right_eye_closed_count > eye_closed_frames_threshold):
                eye_status_text = "Sleeping"
                play_alert()  # Play audio alert to wake the person up!
                was_sleeping = True
            elif (left_eye_closed_count > 0 or right_eye_closed_count > 0):
                eye_status_text = "Blinking"  # One or both eyes temporarily closed
            else:
                eye_status_text = "Eyes Open"  # Both eyes are open

            # ==================== DRAW FACE MESH ====================

            # Draw all facial landmark points and connections on the frame
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

        # ==================== DISPLAY INFORMATION ====================

        # Calculate frames per second (FPS)
        end_time = time.time()
        fps = 1.0 / (end_time - start_time + 1e-6)

        # Display all text information on the frame in green color
        cv2.putText(frame, pitch_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, yaw_text, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, roll_text, (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, orientation_text, (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Eyes: {eye_status_text}", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, img_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Resize frame to full HD and display it
        resized_frame = cv2.resize(frame, (1920, 1080))
        cv2.imshow("Head Pose & Blink Detection", resized_frame)

        # ==================== KEYBOARD CONTROLS ====================

        # Check for key presses (wait 5ms)
        key = cv2.waitKey(5) & 0xFF

        if key in [27, ord('q')]:  # ESC or 'q' key
            break  # Exit the program
        elif key == ord('r'):  # 'r' key
            # Reset head pose offsets to current position
            # This recalibrates "forward" to the current head position
            pitch_offset = display_pitch
            yaw_offset = display_yaw
            roll_offset = display_roll
            print("Pitch, yaw, roll reset to 0.")

    # ==================== CLEANUP ====================

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    # Stop and close the camera
    picam2.close()

# Entry point of the program
if __name__ == "__main__":
    main()
