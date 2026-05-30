import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import time
import queue

# Load the trained model
model = tf.keras.models.load_model("sign_model.h5")
labels = np.load("labels.npy")

print(f"Loaded model! Can recognize: {labels}")

# Setup speech queue
speech_queue = queue.Queue()

def speech_worker():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

# Start speech thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

# Load hand tracker
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Open camera
cap = cv2.VideoCapture(0)

print("Camera started! Show your hand. Press Q to quit.")

current_sign = ""
last_spoken_time = 0
cooldown = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        landmarks = []
        for landmark in result.hand_landmarks[0]:
            landmarks.append([landmark.x, landmark.y, landmark.z])

        for landmark in result.hand_landmarks[0]:
            h, w, _ = frame.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        flat = np.array(landmarks).flatten().reshape(1, -1)
        prediction = model.predict(flat, verbose=0)
        confidence = np.max(prediction)
        current_sign = labels[np.argmax(prediction)]

        if confidence > 0.8:
            cv2.putText(frame, f"Sign: {current_sign.upper()}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.putText(frame, f"Confidence: {confidence * 100:.1f}%",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            now = time.time()
            if (now - last_spoken_time) > cooldown and speech_queue.empty():
                last_spoken_time = now
                speech_queue.put(current_sign)

    else:
        cv2.putText(frame, "Show your hand!",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Sign Language Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
speech_queue.put(None)