import cv2
import face_recognition
import os
import configs


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def saveFaces(frame, face_counter):
    # Convert the frame to grayscale (Haar Cascades work better on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9)
    print(f'faces:{faces}')
    # Loop through the faces detected in the frame
    for (x, y, w, h) in faces:
        # Draw a rectangle around each face (optional)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Crop the face from the frame
        face = frame[y:y+h, x:x+w]

        # Save the cropped face as an image file
        face_filename = f"captured_faces/face_{face_counter}.jpg"
        cv2.imwrite(face_filename, face)
        img = cv2.imread(face_filename)
        cv2.imshow('Detected faces', img)  

def captureFromIPCamera():

    face_counter=0
    cap = cv2.VideoCapture(configs.BACK_GATE_CAM_URL_HIGH)

    counter = 0
    while True:
        counter+=1
        # print(counter)
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = frame[:, :, ::-1]

        if counter%50==0:
            counter = 0
            saveFaces(frame, face_counter)
            # Find faces in the frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            print(face_locations)
            face_counter += 1

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare the detected face to the known face
                matches = face_recognition.compare_faces([known_encoding], face_encoding)
                
                if True in matches:
                    name = "Known Person"
                    color = (0, 255, 0)  # Green for known faces
                else:
                    name = "Unknown"
                    color = (0, 0, 255)  # Red for unknown faces
                print(name)

                # Draw rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display the frame
        # print("before iamshow ")
        cv2.imshow("Video", frame)
        # print("After show")

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

# capture_images('Shakti')
