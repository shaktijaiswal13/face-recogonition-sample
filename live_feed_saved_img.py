import cv2
import os
from datetime import datetime
import configs

rtsp_url = configs.OFFICE_CAM_URL_HIGH

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Unable to connect to the camera stream.")
else:
    print("Connected to the camera stream successfully.")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    save_folder = "captured_faces"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  
        cv2.imshow("Hikvision Stream - Face Detection", frame)
        if len(faces) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_folder, f"face_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved frame with faces as {filename}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
