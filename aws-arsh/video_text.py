import cv2
import face_recognition
import pyttsx3
import os

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', "com.apple.speech.synthesis.voice.samantha")

video_capture = cv2.VideoCapture('test_clip.mp4')

if not video_capture.isOpened():
    engine.say("Error: Could not access the video.")
    engine.runAndWait()
    exit()

previous_face_count = -1
output_dir = 'saved_faces'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

face_detected_once = False

while True:
    ret, frame = video_capture.read()

    if not ret:
        engine.say("Error: Failed to grab a frame.")
        engine.runAndWait()
        break
    face_locations = face_recognition.face_locations(frame)
    num_faces = len(face_locations)
    if num_faces != previous_face_count:
        previous_face_count = num_faces

        if num_faces > 0 and not face_detected_once:
            engine.say(f"I found {num_faces} faces in the video feed.")
            engine.runAndWait()
            for i, face_location in enumerate(face_locations):
                top, right, bottom, left = face_location
                face_image = frame[top:bottom, left:right]
                frame_filename = os.path.join(
                    output_dir, f"face_{video_capture.get(cv2.CAP_PROP_POS_FRAMES)}_{i+1}.jpg")
                cv2.imwrite(frame_filename, face_image)

            face_detected_once = True

    for face_location in face_locations:
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        face_position_message = f"Face at Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}"
        cv2.putText(frame, face_position_message, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Video Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

engine.say("Face detection completed.")
engine.runAndWait()
