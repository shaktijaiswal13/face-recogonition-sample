import cv2
import numpy as np
import json

with open('camera_config.json', 'r') as file:
    config = json.load(file)

resolution = config['video_feed']['resolution']
IN_WIDTH = resolution['width']
IN_HEIGHT = resolution['height']
RESIZE = config['video_feed']['resize']
SAVE = config['video_feed']['save']
RESIZE_WIDTH = config['video_feed']['resize_width']
RESIZE_HEIGHT = config['video_feed']['resize_height']

plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
if plate_cascade.empty():
    print("Error loading cascade classifier.")
    exit()

cap = cv2.VideoCapture("video.mp4")

ret, frame = cap.read()
if not ret:
    print("Failed to grab the first frame.")
    exit()

original_height, original_width = frame.shape[:2]

print(f"Original resolution: {original_width}x{original_height}")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    if RESIZE:
        frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(gray, 1.1, 10)
    print(f"Detected {len(plates)} plates.")
    
    for (x, y, w, h) in plates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Plate Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
