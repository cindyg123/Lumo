import cv2
import mediapipe as mp
import numpy as np
import math
import time
import pyttsx3
import os
import json
from picamera2 import Picamera2

engine = pyttsx3.init()
engine.setProperty('rate', 150)
ALERT_STYLE_FILE = "/home/wrecker888/Downloads/alert_style.json"
DEFAULT_SONG_PATH = "/home/wrecker888/Downloads/annoying_song.mp3"
    
def play_alert():
    print(f"[DEBUG] play_alert called")

    style = {"type": "default", "message": "", "name": "Friend"}
    if os.path.exists(ALERT_STYLE_FILE):
        print(f"[DEBUG] Found ALERT_STYLE_FILE at: {ALERT_STYLE_FILE}")
        with open(ALERT_STYLE_FILE, "r") as f:
            try:
                config = json.load(f)
                if isinstance(config, dict) and config:
                    username, style = next(iter(config.items()))
                    print(f"[DEBUG] Using alert style for user '{username}': {style}")
            except Exception as e:
                print(f"[ERROR] Failed to parse ALERT_STYLE_FILE: {e}")

    name = style.get("name", "Friend")

    if style["type"] == "tts":
        msg = style.get("message", f"{name}, wake up!")
        engine.say(msg)
    elif style["type"] == "song":
        if os.path.exists(DEFAULT_SONG_PATH):
            from playsound import playsound
            playsound(DEFAULT_SONG_PATH)
            return
        else:
            engine.say("Annoying song file not found.")
    else:
        engine.say(f"{name}, wake up!")

    engine.runAndWait()

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

def main():
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

    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (1920, 1080)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()
    time.sleep(1.0)
    picam2.set_controls({"AfMode": 2})

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

    ear_threshold = 0.35
    eye_closed_frames_threshold = 15
    left_eye_closed_count = 0
    right_eye_closed_count = 0

    pitch_offset = 0.0
    yaw_offset = 0.0
    roll_offset = 0.0

    while True:
        frame = picam2.capture_array()
        start_time = time.time()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)

        img_h, img_w, _ = frame.shape

        pitch_text = ""
        yaw_text = ""
        roll_text = ""
        orientation_text = ""
        eye_status_text = ""

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

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
                reference_3d_points,
                face_2d_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success_pnp:
                rmat, _ = cv2.Rodrigues(rotation_vector)
                euler_angles = rotation_matrix_to_euler_angles(rmat) * (180.0 / math.pi)
                pitch, yaw, roll = euler_angles

                display_pitch = pitch - pitch_offset
                display_yaw = yaw - yaw_offset
                display_roll = roll - roll_offset

                pitch_text = f"Pitch: {display_pitch:.2f}"
                yaw_text = f"Yaw:   {display_yaw:.2f}"
                roll_text = f"Roll:  {display_roll:.2f}"

                pitch_up = (display_pitch > 180 and display_pitch < 353)
                pitch_down = (display_pitch < 180 and display_pitch > 7)
                yaw_left = (display_yaw > 7)
                yaw_right = (display_yaw < -7)

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

                nose_idx = landmark_ids_pose.index(1)
                nose_2d = face_2d_points[nose_idx]
                nose_end_3d = np.array([[0, 0, 100]], dtype=np.float64)
                nose_end_2d, _ = cv2.projectPoints(
                    nose_end_3d, rotation_vector, translation_vector,
                    camera_matrix, dist_coeffs
                )
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_end_2d[0][0][0]), int(nose_end_2d[0][0][1]))
                cv2.line(frame, p1, p2, (255, 0, 0), 2)

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

            if (left_eye_closed_count > eye_closed_frames_threshold and
                right_eye_closed_count > eye_closed_frames_threshold):
                eye_status_text = "Sleeping"
                play_alert()
                was_sleeping = True
            elif (left_eye_closed_count > 0 or right_eye_closed_count > 0):
                eye_status_text = "Blinking"
            else:
                eye_status_text = "Eyes Open"

            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

        end_time = time.time()
        fps = 1.0 / (end_time - start_time + 1e-6)

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

        resized_frame = cv2.resize(frame, (1920, 1080))
        cv2.imshow("Head Pose & Blink Detection", resized_frame)
        key = cv2.waitKey(5) & 0xFF
        if key in [27, ord('q')]:
            break
        elif key == ord('r'):
            pitch_offset = display_pitch
            yaw_offset = display_yaw
            roll_offset = display_roll
            print("Pitch, yaw, roll reset to 0.")

    cv2.destroyAllWindows()
    picam2.close()

if __name__ == "__main__":
    main()