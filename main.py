import cv2
import mediapipe as mp
import numpy as np
import threading

import winsound

cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


NOSE_TIP = 1
nose_positions = []
NOD_THRESHOLD = 0.05
NOD_ALERT_PLAYING = False
FRAME_COUNT = 0
ALERT_FRAMES = 15
ALERT_PLAYING = False

def get_ear(landmarks, eye_indices, w, h):
    points = []
    for idx in eye_indices:
        lm = landmarks.landmark[idx]
        points.append((int(lm.x * w), int(lm.y * h)))
    
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    
    ear = (A + B) / (2.0 * C)
    return ear

def play_alert():
    while ALERT_PLAYING:
        winsound.Beep(1000, 500)
def calibrate():
    print("Calibrating... keep eyes open for 3 seconds")
    ears = []
    start = cv2.getTickCount()
    
    while True:
        ret, frame = cap.read()
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        elapsed = (cv2.getTickCount() - start) / cv2.getTickFrequency()
        
        cv2.putText(frame, f"Calibrating... {3 - int(elapsed)}s", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow("Feed", frame)
        cv2.waitKey(1)
        
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                left_ear = get_ear(landmarks, LEFT_EYE, w, h)
                right_ear = get_ear(landmarks, RIGHT_EYE, w, h)
                ears.append((left_ear + right_ear) / 2.0)
        
        if elapsed >= 3:
            break
    
    avg = sum(ears) / len(ears)
    threshold = avg * 0.8
    print(f"Calibration done. Your EAR: {avg:.2f}, Threshold set to: {threshold:.2f}")
    return threshold
THRESHOLD = calibrate()

while True:
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            left_ear = get_ear(landmarks, LEFT_EYE, w, h)
            right_ear = get_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < THRESHOLD:
                FRAME_COUNT += 1
            else:
                FRAME_COUNT = 0

            if FRAME_COUNT >= ALERT_FRAMES:
                cv2.putText(frame, "DROWSY! WAKE UP!", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not ALERT_PLAYING:
                    ALERT_PLAYING = True
                    threading.Thread(target=play_alert, daemon=True).start()
            else:
                ALERT_PLAYING = False

            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            nose = landmarks.landmark[NOSE_TIP]
            nose_positions.append(nose.y)

            if len(nose_positions) > 30:
              nose_positions.pop(0)

            if len(nose_positions) == 30:
                diff = max(nose_positions) - min(nose_positions)
                if diff > NOD_THRESHOLD:
                    cv2.putText(frame, "HEAD NOD DETECTED!", (30, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                    if not NOD_ALERT_PLAYING:
                        NOD_ALERT_PLAYING = True
                        threading.Thread(target=lambda: winsound.Beep(1500, 500), daemon=True).start()
                else:
                    NOD_ALERT_PLAYING = False
    else:
        FRAME_COUNT = 0
        ALERT_PLAYING = False
        cv2.putText(frame, "NO FACE DETECTED", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
