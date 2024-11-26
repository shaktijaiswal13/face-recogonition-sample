import cv2
import face_recognition
import os
import numpy as np
import configs

RECOGNITION_CAM_URL = configs.GALI_CAM_URL_LOW


SAMPLE_COUNT = 100
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
frame_interval = 10


def get_video_source(source):
    if isinstance(source, str):
        return cv2.VideoCapture(source)  # RTSP URL
    elif isinstance(source, int):
        return cv2.VideoCapture(source)  # Camera index
    else:
        raise ValueError("Invalid video source. Provide a valid RTSP URL or camera index.")


# Function to capture images and store in dataset folder
def capture_images(User , source):
    if not os.path.exists('Faces'):
        os.makedirs('Faces')

    cap = get_video_source(source)
    count = 0
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using both frontal and profile face cascades
        faces_frontal = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        faces_profile = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Combine detected faces from both methods
        faces = list(faces_frontal) + list(faces_profile)

        # Only capture images every 'frame_interval' frames
        if frame_counter % frame_interval == 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imwrite(f'Faces/{User }_{count}.jpg', gray[y:y + h, x:x + w])
                count += 1

        cv2.imshow('Capture Faces', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if count >= SAMPLE_COUNT:
            break

        frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()

def train_model(label):
    faces = []
    labels = []

    for file_name in os.listdir('Faces'):
        if file_name.endswith('.jpg'):
            name = file_name.split('_')[0]
            image = cv2.imread(os.path.join('Faces', file_name))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale image
            detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            detected_faces_profile = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Check if a face is detected (frontal or profile)
            if len(detected_faces) > 0:
                face_crop = gray[detected_faces[0][1]:detected_faces[0][1] + detected_faces[0][3],
                                 detected_faces[0][0]:detected_faces[0][0] + detected_faces[0][2]]
                faces.append(face_crop)
                labels.append(label[name])
            elif len(detected_faces_profile) > 0:
                face_crop_profile = gray[detected_faces_profile[0][1]:detected_faces_profile[0][1] + detected_faces_profile[0][3],
                                         detected_faces_profile[0][0]:detected_faces_profile[0][0] + detected_faces_profile[0][2]]
                faces.append(face_crop_profile)
                labels.append(label[name])

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save('trained_model.xml')
    return recognizer


# Function to recognize faces
def recognize_faces(recognizer, label, source):
    if recognizer is None:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trained_model.xml')
    
    cap = get_video_source(source)
    label_name = {value: key for key, value in label.items()}

    while True: 
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using both frontal and profile face cascades
        faces_frontal = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        faces_profile = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Combine detected faces from both methods into a single list
        faces = list(faces_frontal) + list(faces_profile)
        # Recognize and label the faces
        for (x, y, w, h) in faces:
            label, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence > 80 and confidence < 110:  # Adjust threshold as needed
                print(f'label:{label}, confidence:{confidence}')
                cv2.putText(frame, label_name[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Recognize Faces', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    # capture_images('Shakti')
    # capture_images('Neha')
    # capture_images('Riyanshi')
    # capture_images('Arshdeep', rtsp_url)
    # capture_images('Vaishnavi', 0)
    label = {'Shakti':0,'Neha':1,'Riyanshi':2, 'Arshdeep':3 ,'Vaishnavi':4}
    # label = {'Shakti':0, 'Neha':2}
    # label = {'Arshdeep':0}
    # Train the model
    Recognizer =train_model(label)
    # # Recognize the live faces
    recognize_faces(Recognizer, label, RECOGNITION_CAM_URL)
    # recognize_faces(Recognizer, label, 0)  
    # recognize_faces(None, label)
