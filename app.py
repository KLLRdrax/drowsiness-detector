import sys
import cv2
import mediapipe as mp
import numpy as np
import threading
import winsound
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QFrame
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
NOSE_TIP = 1

def get_ear(landmarks, eye_indices, w, h):
    points = []
    for idx in eye_indices:
        lm = landmarks.landmark[idx]
        points.append((int(lm.x * w), int(lm.y * h)))
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    return (A + B) / (2.0 * C)

class DrowsinessApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drowsiness Detector")
        self.setStyleSheet("background-color: #1a1a2e;")

        self.cap = cv2.VideoCapture(0)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()

        self.frame_count = 0
        self.alert_frames = 15
        self.alert_playing = False
        self.drowsy_events = 0
        self.nose_positions = []
        self.nod_threshold = 0.05
        self.threshold = 0.25
        self.session_seconds = 0
        self.break_reminded = False
        self.break_alerted = False

        self.init_ui()
        self.calibrate()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.session_timer = QTimer()
        self.session_timer.timeout.connect(self.update_session)
        self.session_timer.start(1000)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        self.feed_label = QLabel()
        self.feed_label.setFixedSize(640, 480)
        self.feed_label.setStyleSheet("border: 2px solid #16213e; border-radius: 10px;")
        main_layout.addWidget(self.feed_label)

        right = QVBoxLayout()
        right.setSpacing(15)

        title = QLabel("DROWSINESS\nDETECTOR")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #e94560;")
        title.setAlignment(Qt.AlignCenter)
        right.addWidget(title)

        ear_frame = QFrame()
        ear_frame.setStyleSheet("background-color: #16213e; border-radius: 10px; padding: 10px;")
        ear_layout = QVBoxLayout(ear_frame)
        ear_title = QLabel("EAR VALUE")
        ear_title.setStyleSheet("color: #a8a8b3; font-size: 12px;")
        ear_title.setAlignment(Qt.AlignCenter)
        self.ear_label = QLabel("0.00")
        self.ear_label.setFont(QFont("Arial", 32, QFont.Bold))
        self.ear_label.setStyleSheet("color: #00b4d8;")
        self.ear_label.setAlignment(Qt.AlignCenter)
        ear_layout.addWidget(ear_title)
        ear_layout.addWidget(self.ear_label)
        right.addWidget(ear_frame)

        status_frame = QFrame()
        status_frame.setStyleSheet("background-color: #16213e; border-radius: 10px; padding: 10px;")
        status_layout = QVBoxLayout(status_frame)
        status_title = QLabel("STATUS")
        status_title.setStyleSheet("color: #a8a8b3; font-size: 12px;")
        status_title.setAlignment(Qt.AlignCenter)
        self.status_label = QLabel("NORMAL")
        self.status_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.status_label.setStyleSheet("color: #06d6a0;")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(status_title)
        status_layout.addWidget(self.status_label)
        right.addWidget(status_frame)

        counter_frame = QFrame()
        counter_frame.setStyleSheet("background-color: #16213e; border-radius: 10px; padding: 10px;")
        counter_layout = QVBoxLayout(counter_frame)
        counter_title = QLabel("DROWSY EVENTS")
        counter_title.setStyleSheet("color: #a8a8b3; font-size: 12px;")
        counter_title.setAlignment(Qt.AlignCenter)
        self.counter_label = QLabel("0")
        self.counter_label.setFont(QFont("Arial", 32, QFont.Bold))
        self.counter_label.setStyleSheet("color: #e94560;")
        self.counter_label.setAlignment(Qt.AlignCenter)
        counter_layout.addWidget(counter_title)
        counter_layout.addWidget(self.counter_label)
        right.addWidget(counter_frame)

        session_frame = QFrame()
        session_frame.setStyleSheet("background-color: #16213e; border-radius: 10px; padding: 10px;")
        session_layout = QVBoxLayout(session_frame)
        session_title = QLabel("SESSION TIME")
        session_title.setStyleSheet("color: #a8a8b3; font-size: 12px;")
        session_title.setAlignment(Qt.AlignCenter)
        self.session_label = QLabel("00:00")
        self.session_label.setFont(QFont("Arial", 32, QFont.Bold))
        self.session_label.setStyleSheet("color: #06d6a0;")
        self.session_label.setAlignment(Qt.AlignCenter)
        session_layout.addWidget(session_title)
        session_layout.addWidget(self.session_label)
        right.addWidget(session_frame)

        right.addStretch()
        main_layout.addLayout(right)

    def calibrate(self):
        self.status_label.setText("CALIBRATING...")
        self.status_label.setStyleSheet("color: #ffd166;")
        ears = []
        start = cv2.getTickCount()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            elapsed = (cv2.getTickCount() - start) / cv2.getTickFrequency()
            cv2.putText(frame, f"Calibrating... {3 - int(elapsed)}s", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    left_ear = get_ear(landmarks, LEFT_EYE, w, h)
                    right_ear = get_ear(landmarks, RIGHT_EYE, w, h)
                    ears.append((left_ear + right_ear) / 2.0)
            if elapsed >= 3:
                break
        if ears:
            avg = sum(ears) / len(ears)
            self.threshold = avg * 0.8
        self.status_label.setText("NORMAL")
        self.status_label.setStyleSheet("color: #06d6a0;")

    def play_alert(self):
        while self.alert_playing:
            winsound.Beep(1000, 500)

    def update_session(self):
        self.session_seconds += 1
        mins = self.session_seconds // 60
        secs = self.session_seconds % 60
        self.session_label.setText(f"{mins:02d}:{secs:02d}")

        if self.session_seconds == 7200 and not self.break_reminded:
            self.break_reminded = True
            self.status_label.setText("TAKE A BREAK!")
            self.status_label.setStyleSheet("color: #ffd166;")
            winsound.Beep(800, 1000)

        if self.session_seconds == 14400 and not self.break_alerted:
            self.break_alerted = True
            self.status_label.setText("REST NOW!")
            self.status_label.setStyleSheet("color: #e94560;")
            winsound.Beep(800, 2000)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                left_ear = get_ear(landmarks, LEFT_EYE, w, h)
                right_ear = get_ear(landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                self.ear_label.setText(f"{avg_ear:.2f}")

                if avg_ear < self.threshold:
                    self.frame_count += 1
                else:
                    self.frame_count = 0

                if self.frame_count >= self.alert_frames:
                    if not self.alert_playing:
                        self.alert_playing = True
                        self.drowsy_events += 1
                        self.counter_label.setText(str(self.drowsy_events))
                        threading.Thread(target=self.play_alert, daemon=True).start()
                    self.status_label.setText("DROWSY!")
                    self.status_label.setStyleSheet("color: #e94560;")
                else:
                    self.alert_playing = False
                    self.status_label.setText("NORMAL")
                    self.status_label.setStyleSheet("color: #06d6a0;")

                nose = landmarks.landmark[NOSE_TIP]
                self.nose_positions.append(nose.y)
                if len(self.nose_positions) > 30:
                    self.nose_positions.pop(0)
                if len(self.nose_positions) == 30:
                    diff = max(self.nose_positions) - min(self.nose_positions)
                    if diff > self.nod_threshold:
                        self.status_label.setText("HEAD NOD!")
                        self.status_label.setStyleSheet("color: #ffd166;")
        else:
            self.frame_count = 0
            self.alert_playing = False
            self.ear_label.setText("--")
            self.status_label.setText("NO FACE")
            self.status_label.setStyleSheet("color: #a8a8b3;")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qt_image = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.feed_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

app = QApplication(sys.argv)
window = DrowsinessApp()
window.show()
sys.exit(app.exec_())