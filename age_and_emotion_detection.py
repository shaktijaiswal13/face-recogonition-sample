import os
import cv2
import numpy as np
from zipfile import ZipFile
from urllib.request import urlretrieve
import configs
from deepface import DeepFace
from tensorflow.keras.models import load_model
import json
with open('camera_config.json', 'r') as file:
    config = json.load(file)

RECOGNITION_CAM_URL = configs.USB_CAM_URL
# RECOGNITION_CAM_URL = configs.GALI_CAM_URL_HIGH
# RECOGNITION_CAM_URL = configs.SAVED_VIDEO_PATH
# RECOGNITION_CAM_URL = configs.BACK_GATE_CAM_URL_HIGH
# RECOGNITION_CAM_URL = configs.OFFICE_CAM_URL_HIGH

resolution = config['logitech_camera']['resolution']
IN_WIDTH = resolution['width']
IN_HEIGHT = resolution['height']
RESIZE = config['logitech_camera']['resize']
SAVE = config['logitech_camera']['save']
RESIZE_WIDTH = config['logitech_camera']['resize_width']
RESIZE_HEIGHT = config['logitech_camera']['resize_height']

# model = load_model('gender_detection.model')
# model.save('gender_detection.h5')

gender_model = load_model('gender_detection.h5')

# ======================== Downloading Assets ========================
def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assets....", end="")
    urlretrieve(url, save_path)
    try:
        with ZipFile(save_path) as z:
            z.extractall(os.path.split(save_path)[0])
        print("Done")
    except Exception as e:
        print("\nInvalid file.", e)

asset_zip_path = os.path.join(os.getcwd(), f"opencv_bootcamp_assets_12.zip")

# Download if assets ZIP does not exists
if not os.path.exists(asset_zip_path):
    download_and_unzip(configs.OPENCV_BOOTCAMP_ASSETS_URL, asset_zip_path)

# Folder to save detected faces
faces_folder = "faces"
if not os.path.exists(faces_folder):
    os.makedirs(faces_folder)

# Prepare the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load saved faces for recognition
face_images = []  
face_labels = []  
face_names = []   

# Load the images from 'faces' folder and train the recognizer
face_count = 0
for filename in os.listdir(faces_folder):
    if filename.endswith(".jpg"):
        face_image = cv2.imread(os.path.join(faces_folder, filename), cv2.IMREAD_GRAYSCALE)
        if face_image is not None:
            face_images.append(face_image)
            face_labels.append(face_count)
            face_names.append(filename.split('_')[0])
            face_count += 1

if len(face_images) > 0:
    recognizer.train(face_images, np.array(face_labels))
    print(f"Recognizer trained with {len(face_images)} faces.")
    recognizer.save('face_recognizer.xml')
else:
    print("No faces found to train.")


def preprocess_for_keras_model(image):
    image = cv2.resize(image, (96, 96)) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float32")
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image


# ========================= Face Recognition and Age Estimation =====================

def recognize_face():
    print(f'recognize_face started with url: {RECOGNITION_CAM_URL}')
    
    # Initialize video capture
    source = cv2.VideoCapture(RECOGNITION_CAM_URL)

    if not source.isOpened():
        print("Error: Unable to open camera.")
        return

    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
    AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']


    while True:
        has_frame, frame = source.read()
        if not has_frame:
            print('Breaking - No frame captured')
            break
        
        try:
            if RESIZE:
                frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
            frame_height = frame.shape[0]
            frame_width = frame.shape[1]

            # ===================== Face Detection and Recognition ========================
            blob = cv2.dnn.blobFromImage(frame, 1.0, (IN_WIDTH, IN_HEIGHT), (104, 117, 123), swapRB=False, crop=False)
            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                detection_confidence = detections[0, 0, i, 2]
                if detection_confidence > configs.DETECTION_CONFIDENCE_THRESHOLD:
                    x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
                    y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
                    x_right_top = int(detections[0, 0, i, 5] * frame_width)
                    y_right_top = int(detections[0, 0, i, 6] * frame_height)

                    cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))

                    detected_face = frame[y_left_bottom:y_right_top, x_left_bottom:x_right_top]

                    detected_face_gray = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
                    label, prediction_confidence = recognizer.predict(detected_face_gray)
                    if prediction_confidence > configs.PREDICTION_CONFIDENCE and prediction_confidence < 100:
                        person = face_names[label]
                        # print(f"Recognized: {person}")
                    else:
                        person = "Unknown"

                    # Age detection
                    blob = cv2.dnn.blobFromImage(detected_face, 1.0, (227, 227), (78, 87, 114), swapRB=False)
                    age_net.setInput(blob)
                    age_predictions = age_net.forward()
                    age = AGE_LIST[age_predictions[0].argmax()]
                    
                    result = DeepFace.analyze(detected_face, actions=['emotion'], enforce_detection=False)
                    emotion = result[0]['dominant_emotion']
                    
                    preprocessed_face = preprocess_for_keras_model(detected_face)
                    gender_predictions = gender_model.predict(preprocessed_face)
                    gender = 'Male' if gender_predictions[0][0] > 0.5 else 'Female'


                    cv2.putText(frame, f"{gender} - {age} - {emotion}", (x_left_bottom, y_left_bottom-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Camera Preview", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as ex:
            print(ex)

    source.release()
    cv2.destroyWindow("Camera Preview")

# ========================= Main function =====================

if __name__ == "__main__":
    print("__main__ started.")

    recognize_face()
