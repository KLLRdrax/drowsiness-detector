# 🚗 Driver Drowsiness Detection System

A real-time driver drowsiness detection system built with Python, OpenCV, and MediaPipe. Monitors eye movement and head position to alert drivers when signs of fatigue are detected.

## Features

- 👁️ Real-time eye tracking using MediaPipe Face Mesh
- 📊 Per-user EAR (Eye Aspect Ratio) calibration on startup
- 🔔 Audio alert when drowsiness is detected
- 🤚 Head nod detection
- ⏱️ Session timer with government-recommended break reminders
- 🖥️ Clean PyQt5 desktop UI

## How It Works

The system calculates the **Eye Aspect Ratio (EAR)** using facial landmarks:
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
When EAR drops below the calibrated threshold for 15 consecutive frames, the driver is flagged as drowsy and an audio alert fires.

## Installation

```bash
pip install opencv-python mediapipe numpy PyQt5
```

## Run

```bash
python app.py
```

Look at the camera for 3 seconds during calibration, then drive safe.

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- PyQt5