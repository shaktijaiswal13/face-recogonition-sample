import os
import cv2
import sys
import numpy as np
import configs

# RECOGNITION_CAM_URL = configs.INTERNAL_CAM_URL
RECOGNITION_CAM_URL = configs.GALI_CAM_URL_HIGH
# RECOGNITION_CAM_URL = configs.SAVED_VIDEO_PATH
# RECOGNITION_CAM_URL = configs.FRONT_CAM_URL_HIGH

RESIZE = True
SAVE = False
IN_WIDTH = 500
IN_HEIGHT = 500
RESIZE_WIDTH = 800
RESIZE_HEIGHT = 600

# ========================-Downloading Assets-========================
def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assets....", end="")

    # Downloading zip file using urllib package.
    urlretrieve(url, save_path)

    try:
        # Extracting zip file using the zipfile package.
        with ZipFile(save_path) as z:
            # Extract ZIP file contents in the same directory.
            z.extractall(os.path.split(save_path)[0])

        print("Done")

    except Exception as e:
        print("\nInvalid file.", e)

asset_zip_path = os.path.join(os.getcwd(), f"opencv_bootcamp_assets_12.zip")

# Download if assets ZIP does not exists.
if not os.path.exists(asset_zip_path):
    download_and_unzip(configs.OPENCV_BOOTCAMP_ASSETS_URL, asset_zip_path)
# ====================================================================

# Define common parameters
win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

# Load the images from the 'faces' folder and train the recognizer
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

# ====================================================================
# Recognize face function
def recognize_face(label):
    source = cv2.VideoCapture(RECOGNITION_CAM_URL)
    label_name = {value: key for key, value in label.items()}
    saved_faces_dir = "detected_faces"  
    frame = None
    while cv2.waitKey(1) != 27:
        has_frame, frame = source.read()
        if not has_frame:
            print('breaking')
            break
        try:
            if RESIZE:
                frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
            frame_height = frame.shape[0]
            frame_width = frame.shape[1]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            detected_face = None
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))  
                
                detected_face = frame[y:y+h, x:x+w]

                if detected_face is not None:
                    detected_face_gray = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
                    label, prediction_confidence = recognizer.predict(detected_face_gray)

                    if prediction_confidence > configs.PREDICTION_CONFIDENCE and prediction_confidence < 100:
                        person = face_names[label]
                    else:
                        person = ""

                    if SAVE:
                        face_filename = os.path.join(saved_faces_dir, f"{person}_{cv2.getTickCount()}.jpg")
                        cv2.imwrite(face_filename, detected_face)

                    label_text = f"{person}, Predic: {prediction_confidence:.2f}%"
                    print(label_text)
                    cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(win_name, frame)

        except Exception as ex:
            print(ex)
    source.release()
    cv2.destroyWindow(win_name)

# ====================================================================
# Main function
if __name__ == "__main__":
    label = {'Shakti': 0, 'Neha': 1, 'Riyanshi': 2, 'Arshdeep': 3, 'Vaishnavi': 4}
    recognize_face(label)
