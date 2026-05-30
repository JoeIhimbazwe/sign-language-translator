import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import json

# Load hand tracker
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Create data folder
if not os.path.exists("data"):
    os.makedirs("data")

# Ask what sign to record
sign_name = input("Enter the sign name you want to record (e.g. hello, yes, no): ")
sign_folder = f"data/{sign_name}"

if not os.path.exists(sign_folder):
    os.makedirs(sign_folder)

print(f"\nRecording sign: {sign_name}")
print("Press SPACE to save a sample, Q to quit")
print("Try to collect at least 30 samples!\n")

sample_count = 0
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    # Draw landmarks
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            for landmark in hand:
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    # Show sample count on screen
    cv2.putText(frame, f"Sign: {sign_name} | Samples: {sample_count}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "SPACE = save sample | Q = quit", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save sample when SPACE is pressed
    if key == ord(' '):
        if result.hand_landmarks:
            landmarks = []
            for landmark in result.hand_landmarks[0]:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            
            filename = f"{sign_folder}/sample_{sample_count}.json"
            with open(filename, 'w') as f:
                json.dump(landmarks, f)
            
            sample_count += 1
            print(f"Sample {sample_count} saved!")
        else:
            print("No hand detected! Show your hand to the camera.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nDone! Saved {sample_count} samples for sign: {sign_name}")