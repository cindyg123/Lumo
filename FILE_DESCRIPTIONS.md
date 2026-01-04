# SafeSense - File Descriptions

## Overview
SafeSense is a drowsiness detection system designed to monitor drivers and alert them when signs of fatigue are detected. The system uses computer vision and facial landmark detection to track head pose and eye closure.

---

## Files

### [web_dashboard.py](web_dashboard.py)
**Flask Web Application Server**

A comprehensive web server that provides the user interface and management system for SafeSense.

**Features:**
- User authentication (signup/login)
- Account management with personalized names
- Alert style customization (default wake-up, custom TTS messages, or annoying song)
- Emergency contacts management (10-digit phone numbers)
- Google Maps integration for finding nearby rest stops
- JSON-based data persistence for users and alert configurations

**Key Routes:**
- `/signup` - User registration
- `/login` - User authentication
- `/main` - Main dashboard
- `/setname` - Set account display name
- `/alertstyle` - Configure alert preferences
- `/contacts` - Manage emergency contacts
- `/reststops` - Find nearby convenience stores via Google Maps

**Data Files:**
- `users.json` - Stores user credentials, names, and emergency contacts
- `alert_style.json` - Stores alert preferences per user

---

### [safesense.py](safesense.py)
**Production Drowsiness Detection System**

The production-ready version of the SafeSense drowsiness detection system with integrated audio alerts.

**Features:**
- Real-time face mesh tracking using MediaPipe
- Head pose estimation (pitch, yaw, roll) using PnP algorithm
- Eye Aspect Ratio (EAR) calculation for blink/drowsiness detection
- Visual direction detection (looking up/down/left/right)
- Audio alerts via pyttsx3 (text-to-speech)
- Customizable alert system that reads from `alert_style.json`
- Raspberry Pi Camera (Picamera2) support at 1920x1080 resolution
- FPS monitoring and display

**Detection Logic:**
- EAR threshold: 0.35
- Frames threshold: 15 consecutive frames with closed eyes triggers "Sleeping" state
- Automatically calls `play_alert()` when sleeping is detected

**Key Functions:**
- `rotation_matrix_to_euler_angles()` - Converts rotation matrix to Euler angles
- `calculate_ear()` - Calculates Eye Aspect Ratio for drowsiness detection
- `play_alert()` - Triggers customizable audio alerts based on user preferences

---

### [basic_drowsiness_detector.py](basic_drowsiness_detector.py)
**Basic Drowsiness Detection System**

A streamlined version of the drowsiness detection system without audio alerts.

**Features:**
- Face mesh tracking with MediaPipe
- Head pose estimation (pitch, yaw, roll)
- Eye closure detection using EAR
- Visual orientation feedback
- Raspberry Pi Camera support
- Lightweight and focused on core detection

**Differences from safesense.py:**
- No audio alert system
- No pyttsx3 or playsound dependencies
- Displays at 640x480 (vs 1920x1080)
- Pure detection without notification features

---

### [head_pose_prototype.py](head_pose_prototype.py)
**Initial Prototype Implementation**

The earliest version of the head pose estimation system.

**Features:**
- Basic head pose tracking using MediaPipe Face Mesh
- 6 facial landmark tracking (eyes and nose)
- Real-time orientation detection
- Webcam support (standard USB/built-in camera)
- PnP-based pose estimation

**Key Characteristics:**
- Uses standard webcam (`cv2.VideoCapture(0)`)
- Simpler landmark set (6 points vs 7 in later versions)
- Basic orientation text display
- No eye closure detection
- Foundation for later SafeSense versions

---

### [quantum_eye_classifier.py](quantum_eye_classifier.py)
**Quantum Neural Network for Eye State Classification**

An experimental machine learning model using quantum computing concepts for eye open/closed detection.

**Features:**
- Hybrid quantum-classical neural network using PennyLane
- TensorFlow/Keras integration
- Google Colab compatible
- Image augmentation with noise
- Dataset handling for open/closed eye images

**Model Architecture:**
- Quantum layer with 3 qubits using angle embedding
- Classical dense layers (128 units + output layer)
- Rotation gates (RX) and CNOT entanglement
- Softmax output for classification

**Training Configuration:**
- Image size: 28x28 pixels
- Batch size: 32
- 80/20 train/test split
- 10 epochs
- Adam optimizer with sparse categorical crossentropy loss

**Note:** This appears to be an experimental/research component and may not be integrated into the main SafeSense application.

---

## System Architecture

```
┌─────────────────────────┐
│  web_dashboard.py       │ ← Web UI & User Management
└────────┬────────────────┘
         │
         ├── users.json
         └── alert_style.json
                 ↓
         ┌──────────────┐
         │  safesense.py│ ← Main Detection System
         └──────────────┘
                 ↑
         ┌──────────────┐
         │  Picamera2   │ ← Raspberry Pi Camera
         └──────────────┘
```

---

## Dependencies

**Core Detection:**
- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)
- Picamera2 (Raspberry Pi)

**Audio Alerts:**
- pyttsx3 (text-to-speech)
- playsound (music playback)

**Web Server:**
- Flask
- Google Maps API

**Quantum ML (quantum_eye_classifier.py):**
- PennyLane
- TensorFlow/Keras
- Matplotlib

---

## Configuration Files

### users.json
```json
{
  "username": {
    "password": "...",
    "name": "Display Name",
    "contacts": ["1234567890", "0987654321"]
  }
}
```

### alert_style.json
```json
{
  "username": {
    "type": "tts|default|song",
    "message": "Custom wake up message",
    "name": "User's Name"
  }
}
```

---

## Usage

1. **Start the web server:**
   ```bash
   python web_dashboard.py
   ```
   Access at `http://localhost:5000`

2. **Run the detection system:**
   ```bash
   python safesense.py
   ```

3. **Controls:**
   - Press `q` or `ESC` to quit
   - Press `r` to reset pitch/yaw/roll offsets to current position

---

## Project Evolution

1. **head_pose_prototype.py** - Initial head pose tracking concept
2. **basic_drowsiness_detector.py** - Added eye closure detection
3. **safesense.py** - Production system with audio alerts and customization
4. **web_dashboard.py** - Web-based configuration and management
5. **quantum_eye_classifier.py** - Experimental quantum ML approach (research)
