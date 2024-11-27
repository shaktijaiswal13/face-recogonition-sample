from PIL import Image
import face_recognition
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('voice', "com.apple.speech.synthesis.voice.samantha")

image = face_recognition.load_image_file("test.jpeg")
face_locations = face_recognition.face_locations(image)
num_faces = len(face_locations)
engine.say(f"I found {num_faces} face(s) in this photograph.")
engine.runAndWait()

for face_location in face_locations:
    top, right, bottom, left = face_location
    face_position_message = f"A face is located at pixel location Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}"
    print(face_position_message)

    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()

engine.say("Face detection completed.")
engine.runAndWait()
