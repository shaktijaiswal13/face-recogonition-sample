import cv2
import face_recognition
import os
import numpy as np
import configs

TRAINING_CAM_URL = configs.INTERNAL_CAM_URL

IN_WIDTH = 500
IN_HEIGHT = 500
SAMPLE_COUNT = 100
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
frame_interval = 10

# Model parameters
in_width = IN_WIDTH
in_height = IN_HEIGHT
mean = [104, 117, 123]


# Define common parameters
win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
# Load the pre-trained face detection model (Caffe-based)
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
# Folder to save detected faces
faces_folder = "faces"
if not os.path.exists(faces_folder):
    os.makedirs(faces_folder)
# ====================================================================
# Save image function
def save_image(person_name):
    source = cv2.VideoCapture(TRAINING_CAM_URL)
    face_count = 0

    while cv2.waitKey(1) != 27:
        has_frame, frame = source.read()
        if not has_frame:
            break
        # frame = cv2.flip(frame, 1)
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > configs.DETECTION_CONFIDENCE_THRESHOLD:
                x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
                y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
                x_right_top = int(detections[0, 0, i, 5] * frame_width)
                y_right_top = int(detections[0, 0, i, 6] * frame_height)

                cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))
                # label = "Confidence: %.4f" % configs.CONFIDENCE

                face_img = frame[y_left_bottom:y_right_top, x_left_bottom:x_right_top]
                face_filename = os.path.join(faces_folder, f"{person_name }_{face_count}.jpg")
                cv2.imwrite(face_filename, face_img)
                face_count += 1  

        cv2.imshow(win_name, frame)

    source.release()
    cv2.destroyWindow(win_name)

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

if __name__=="__main__":
    # capture_images('Shakti')
    # capture_images('Neha')
    # capture_images('Riyanshi')
    # capture_images('Arshdeep', rtsp_url)
    # capture_images('Vaishnavi', 0)
    label = {'Shakti':0,'Neha':1,'Riyanshi':2, 'Arshdeep':3 ,'Vaishnavi':4, 'Mummy':5}
    # label = {'Shakti':0, 'Neha':2}
    # label = {'Arshdeep':0}
    save_image('Riyanshi')
