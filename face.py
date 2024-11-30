import cv2

# Define Models
face_pbtxt = "models/opencv_face_detector.pbtxt"
face_pb = "models/opencv_face_detector_uint8.pb"
gender_prototxt = "models/gender_deploy.prototxt"
gender_model = "models/gender_net_2.caffemodel"
MODEL_MEAN_VALUES = [104, 117, 123]

# Load Models
face = cv2.dnn.readNet(face_pb, face_pbtxt)
gen = cv2.dnn.readNet(gender_model, gender_prototxt)


# Setup Classifications
gender_classifications = ['Male', 'Female']

# Open webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Check if webcam is opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Resize frame
    img_cp = cv2.resize(frame, (720, 640))

    # Get Image Dimensions & Blob
    img_h = img_cp.shape[0]
    img_w = img_cp.shape[1]
    blob = cv2.dnn.blobFromImage(img_cp, 1.0, (300, 300), MODEL_MEAN_VALUES, True, False)

    face.setInput(blob)
    detected_faces = face.forward()

    face_bounds = []

    # Draw Rectangle Over Faces
    for i in range(detected_faces.shape[2]):
        confidence = detected_faces[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detected_faces[0, 0, i, 3] * img_w)
            y1 = int(detected_faces[0, 0, i, 4] * img_h)
            x2 = int(detected_faces[0, 0, i, 5] * img_w)
            y2 = int(detected_faces[0, 0, i, 6] * img_h)
            cv2.rectangle(img_cp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            face_bounds.append([x1, y1, x2, y2])


    for face_bound in face_bounds:
        try:
            # Crop the face region from the frame
            face_region = img_cp[max(0, face_bound[1] - 15):min(face_bound[3] + 15, img_cp.shape[0] - 1),
                                 max(0, face_bound[0] - 15):min(face_bound[2] + 15, img_cp.shape[1] - 1)]
            
            # Prepare the face region for DNN
            blob = cv2.dnn.blobFromImage(face_region, 1.0, (227, 227), MODEL_MEAN_VALUES, True)

            # Gender prediction
            gen.setInput(blob)
            gender_prediction = gen.forward()
            gender = gender_classifications[gender_prediction[0].argmax()]
            
            # Display age and gender
            cv2.putText(img_cp, f'{gender}', (face_bound[0], face_bound[1] - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")
            continue

    # Display the resulting frame
    cv2.imshow('Webcam Result', img_cp)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
