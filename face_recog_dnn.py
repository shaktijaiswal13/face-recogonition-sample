import os
import cv2
import sys
from zipfile import ZipFile
from urllib.request import urlretrieve
import numpy as np
import configs
import cProfile
import threading
import queue
import json
with open('camera_config.json', 'r') as file:
    config = json.load(file)

# RECOGNITION_CAM_URL = configs.USB_CAM_URL
# RECOGNITION_CAM_URL = configs.BACK_GATE_CAM_URL_HIGH
RECOGNITION_CAM_URL = configs.GALI_CAM_URL_HIGH
# RECOGNITION_CAM_URL = configs.SAVED_VIDEO_PATH
# RECOGNITION_CAM_URL = configs.OFFICE_CAM_URL_HIGH

resolution = config['windows_camera']['resolution']
IN_WIDTH = resolution['width']
IN_HEIGHT = resolution['height']
RESIZE = config['windows_camera']['resize']
SAVE = config['windows_camera']['save']
RESIZE_WIDTH = config['windows_camera']['resize_width']
RESIZE_HEIGHT = config['windows_camera']['resize_height']


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

# # Define common parameters
# win_name = "Camera Preview"
# cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# # Load the pre-trained face detection model (Caffe-based)
# net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
# # Model parameters
in_width = IN_WIDTH
in_height = IN_HEIGHT
mean = [104, 117, 123]

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


def process_frames(frame_queue):
    print('process_frames started')
    while True:
        frameObj = frame_queue.get()
        if frameObj is None:
            print('frameObj is none')
            continue
        print('Processing frameObj')
        resizeAndSaveFrame(frameObj['frame'],frameObj['person'])


if len(face_images) > 0:
    recognizer.train(face_images, np.array(face_labels))
    print(f"Recognizer trained with {len(face_images)} faces.")
    recognizer.save('face_recognizer.xml')
else:
    print("No faces found to train.")
    

def resizeAndSaveFrame(originalFrame, person):
    try:
        saved_faces_dir = "detected_faces"  
        frame_height = originalFrame.shape[0]
        frame_width = originalFrame.shape[1]
        in_width = 600
        in_height = 600
        # frame_height = 600
        # frame_width = 600
        # print(f'frame_width:{frame_width},frame_height:{frame_height}')
        blob = cv2.dnn.blobFromImage(originalFrame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
        net.setInput(blob)
        detections = net.forward()
        detected_face = None
        for i in range(detections.shape[2]):
            detection_confidence = detections[0, 0, i, 2]
            if detection_confidence > configs.DETECTION_CONFIDENCE_THRESHOLD:
                x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
                y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
                x_right_top = int(detections[0, 0, i, 5] * frame_width)
                y_right_top = int(detections[0, 0, i, 6] * frame_height)
                if x_left_bottom >= 0 and y_left_bottom >= 0 and x_right_top <= frame_width and y_right_top <= frame_height:
                    detected_face = originalFrame[y_left_bottom:y_right_top, x_left_bottom:x_right_top]

                    if detected_face is not None:
                        # cv2.imshow("Resized Face", detected_face) 
                        face_filename = os.path.join(saved_faces_dir, f"{person}_{cv2.getTickCount()}.jpg")
                        cv2.imwrite(face_filename, detected_face)
    except Exception as e:
        print(e)

# ====================================================================
# Recognize face function
def recognize_face(frame_queue):
    print(f'recognize_face started with url: {RECOGNITION_CAM_URL}')

    frame = None
    while cv2.waitKey(1) != 27:
        has_frame, frame = source.read()
        if not has_frame:
            print('breaking')
            break
        try:
            originalFrame = frame
            if RESIZE:
                frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
            frame_height = frame.shape[0]
            frame_width = frame.shape[1]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
            net.setInput(blob)
            detections = net.forward()

            detected_face = None
            for i in range(detections.shape[2]):
                detection_confidence = detections[0, 0, i, 2]
                if detection_confidence > configs.DETECTION_CONFIDENCE_THRESHOLD:
                    x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
                    y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
                    x_right_top = int(detections[0, 0, i, 5] * frame_width)
                    y_right_top = int(detections[0, 0, i, 6] * frame_height)

                    cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))
        #             # label = "Confidence: %.4f" % confidence
                    
                    if x_left_bottom >= 0 and y_left_bottom >= 0 and x_right_top <= frame_width and y_right_top <= frame_height:
                        detected_face = frame[y_left_bottom:y_right_top, x_left_bottom:x_right_top]

                        if detected_face is not None:
                            person = ""
                            
                            detected_face_gray = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
                            label, prediction_confidence = recognizer.predict(detected_face_gray)
                            if prediction_confidence > configs.PREDICTION_CONFIDENCE and  prediction_confidence < 100:
                                person = face_names[label]
    
                                
                            if SAVE:
                                # frame_queue.put({'frame':originalFrame,'person':person})
                                resizeAndSaveFrame(originalFrame, person)
                                # desired_size = (300, 300)  # Set the size to which you want to increase
                                # resized_face = cv2.resize(detected_face, desired_size)  
                                # cv2.imshow("Resized Face", resized_face) 
                                # face_filename = os.path.join(saved_faces_dir, f"{person}_{cv2.getTickCount()}.jpg")
                                # cv2.imwrite(face_filename, resized_face)

                            # cv2.putText(frame, person, (x_left_bottom, y_left_bottom - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            label_text = f"{person},Predic: { prediction_confidence:.2f}% - Detect: {detection_confidence:.4f}"
                            print(label_text)
                            # cv2.putText(frame, label_text, (x_left_bottom, y_left_bottom - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            t, _ = net.getPerfProfile()
            label = "Inference time: %.2f ms" % (t * 1000.0 / cv2.getTickFrequency())
            # print(label)
            cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            cv2.imshow(win_name, frame)
        except Exception as ex:
            print(ex)
    source.release()
    cv2.destroyWindow(win_name)

# ====================================================================
# Main function
if __name__ == "__main__":
    print("__main__ started.")
    win_name = "Camera Preview"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Load the pre-trained face detection model (Caffe-based)
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    source = cv2.VideoCapture(RECOGNITION_CAM_URL)
    # label_name = {value: key for key, value in label.items()}
    # print(label_name)
    saved_faces_dir = "detected_faces"  
    label = {'Shakti':0,'Neha':1,'Riyanshi':2, 'Arshdeep':3 ,'Vaishnavi':4}
    # save_image('Shakti')
    # cProfile.run('recognize_face(label)')
    # frame_queue = queue.Queue()
    # consumer_thread = threading.Thread(target=process_frames, args=(frame_queue ,))
    # producer_thread = threading.Thread(target=recognize_face, args=(frame_queue ,))
    print("All threads created.")
    # producer_thread.start()
    # consumer_thread.start()
    recognize_face(None)
    # print("All threads started.")
    # Wait for the producer to finish
    # producer_thread.join()

    # Add a None value to signal the consumer to stop
    # frame_queue.put(None)

    # Wait for the consumer to finish
    # consumer_thread.join()
    # print("All threads finished.")
